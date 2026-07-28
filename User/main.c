#include "system.h"
#include "SysTick.h"
#include "usart.h"
#include "led.h"
#include "tftlcd.h"
#include "key.h"
#include "touch.h"

// 替换为新的简洁权重文件
#include "conv_weights_emnist.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// EMNIST Balanced 47 分类标准映射表
// 0-9: 数字, 10-35: 大写字母, 36-46: 小写字母 (部分与大写形似的被合并)
const char EMNIST_MAP[47] = {
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
};

// 结构体定义
typedef struct {
    u16 x;
    u16 y;
} track_point;

track_point ncr_input_buf[600];

// MNIST 中心重影
void center_of_mass_shift(float* grid) {
    float sum_x = 0, sum_y = 0, total_mass = 0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float val = grid[y * 28 + x];
            if (val > 0) {
                sum_x += x * val;
                sum_y += y * val;
                total_mass += val;
            }
        }
    }
    if (total_mass == 0) return;
    int com_x = (int)(sum_x / total_mass + 0.5f);
    int com_y = (int)(sum_y / total_mass + 0.5f);
    int shift_x = 14 - com_x;
    int shift_y = 14 - com_y;
    if (shift_x == 0 && shift_y == 0) return;
    static float temp_grid[784];
    memset(temp_grid, 0, 784 * sizeof(float));
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int new_x = x + shift_x;
            int new_y = y + shift_y;
            if (new_x >= 0 && new_x < 28 && new_y >= 0 && new_y < 28) {
                temp_grid[new_y * 28 + new_x] = grid[y * 28 + x];
            }
        }
    }
    memcpy(grid, temp_grid, 784 * sizeof(float));
}

void draw_hr_brush(int cx, int cy, uint8_t* hr_grid) {
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int px = cx + dx;
            int py = cy + dy;
            if (px >= 0 && px < 84 && py >= 0 && py < 84) {
                hr_grid[py * 84 + px] = 1;
            }
        }
    }
}

// 预处理轨迹
void preprocess_trajectory(track_point* buf, u16 count, float* nn_input)
{
    memset(nn_input, 0, 784 * sizeof(float));
    if (count == 0) return;
    u16 min_x = 0xFFFF, max_x = 0, min_y = 0xFFFF, max_y = 0;
    u16 valid_points = 0;
    for (u16 i = 0; i < count; i++) {
        if (buf[i].x == 0xFFFF) continue;
        if (buf[i].x < min_x) min_x = buf[i].x;
        if (buf[i].x > max_x) max_x = buf[i].x;
        if (buf[i].y < min_y) min_y = buf[i].y;
        if (buf[i].y > max_y) max_y = buf[i].y;
        valid_points++;
    }
    if(valid_points == 0) return;
    float width = max_x - min_x;
    float height = max_y - min_y;
    float max_span = (width > height) ? width : height;
    if (max_span == 0) max_span = 1;
    float scale = 60.0f / max_span;
    float offset_x = 42.0f - (width * scale) / 2.0f;
    float offset_y = 42.0f - (height * scale) / 2.0f;
    static uint8_t hr_grid[7056];
    memset(hr_grid, 0, sizeof(hr_grid));
    int prev_gx = -1, prev_gy = -1;
    for (u16 i = 0; i < count; i++) {
        if (buf[i].x == 0xFFFF) { prev_gx = -1; continue; }
        int gx = (int)((buf[i].x - min_x) * scale + offset_x);
        int gy = (int)((buf[i].y - min_y) * scale + offset_y);
        if (gx < 0) gx = 0; if (gx > 83) gx = 83;
        if (gy < 0) gy = 0; if (gy > 83) gy = 83;
        if (prev_gx != -1) {
            int dx = abs(gx - prev_gx), sx = prev_gx < gx ? 1 : -1;
            int dy = -abs(gy - prev_gy), sy = prev_gy < gy ? 1 : -1;
            int err = dx + dy, e2, cx = prev_gx, cy = prev_gy;
            while (1) {
                draw_hr_brush(cx, cy, hr_grid);
                if (cx == gx && cy == gy) break;
                e2 = 2 * err;
                if (e2 >= dy) { err += dy; cx += sx; }
                if (e2 <= dx) { err += dx; cy += sy; }
            }
        } else { draw_hr_brush(gx, gy, hr_grid); }
        prev_gx = gx; prev_gy = gy;
    }
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int sum = 0;
            for (int dy = 0; dy < 3; dy++) {
                for (int dx = 0; dx < 3; dx++) {
                    if (hr_grid[(y * 3 + dy) * 84 + (x * 3 + dx)]) sum++;
                }
            }
            nn_input[y * 28 + x] = sum / 9.0f;
        }
    }
    center_of_mass_shift(nn_input);
}

// 核心CNN推理逻辑 (47类 EMNIST, 16/32 filters)
int simple_cnn_predict(float *input)
{
    static float conv1_out[16 * 26 * 26];
    static float conv2_out[32 * 11 * 11];

    // 1. Conv1 (1->16, 3x3) — INT8 weights
    for (int c = 0; c < 16; c++) {
        for (int y = 0; y < 26; y++) {
            for (int x = 0; x < 26; x++) {
                float sum = 0;
                for (int dy = 0; dy < 3; dy++) {
                    for (int dx = 0; dx < 3; dx++) {
                        sum += input[(y+dy)*28 + (x+dx)] * (float)conv1_w[c*9 + dy*3 + dx];
                    }
                }
                sum = sum * conv1_w_scale + conv1_b[c];
                conv1_out[(c*26*26) + (y*26) + x] = (sum > 0) ? sum : 0;
            }
        }
    }

    // 2. Pool1 (2x2) -> 16x13x13
    static float pool1_out[16 * 13 * 13];
    for (int c = 0; c < 16; c++) {
        for (int y = 0; y < 13; y++) {
            for (int x = 0; x < 13; x++) {
                float max_v = -1e20f;
                for (int dy=0; dy<2; dy++) for (int dx=0; dx<2; dx++) {
                    float v = conv1_out[(c*26*26) + (y*2+dy)*26 + (x*2+dx)];
                    if (v > max_v) max_v = v;
                }
                pool1_out[(c*13*13) + y*13 + x] = max_v;
            }
        }
    }

    // 3. Conv2 (16->32, 3x3) — INT8 weights
    for (int c = 0; c < 32; c++) {
        for (int y = 0; y < 11; y++) {
            for (int x = 0; x < 11; x++) {
                float sum = 0;
                for (int in_c = 0; in_c < 16; in_c++) {
                    for (int dy = 0; dy < 3; dy++) {
                        for (int dx = 0; dx < 3; dx++) {
                            sum += pool1_out[(in_c*13*13) + (y+dy)*13 + (x+dx)] * (float)conv2_w[c*16*9 + in_c*9 + dy*3 + dx];
                        }
                    }
                }
                sum = sum * conv2_w_scale + conv2_b[c];
                conv2_out[(c*11*11) + (y*11) + x] = (sum > 0) ? sum : 0;
            }
        }
    }

    // 4. Pool2 (2x2) -> 32x5x5
    static float pool2_out[32 * 5 * 5];
    for (int c = 0; c < 32; c++) {
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                float max_v = -1e20f;
                for (int dy=0; dy<2; dy++) for (int dx=0; dx<2; dx++) {
                    float v = conv2_out[(c*11*11) + (y*2+dy)*11 + (x*2+dx)];
                    if (v > max_v) max_v = v;
                }
                pool2_out[(c*5*5) + y*5 + x] = max_v;
            }
        }
    }

    // 5. FC (800 -> 47) — INT8 weights
    int best_idx = 0;
    float max_score = -1e20f;
    for (int j = 0; j < 47; j++) {
        float sum = 0;
        for (int i = 0; i < 800; i++) {
            sum += pool2_out[i] * (float)fc_w[i*47 + j];
        }
        sum = sum * fc_w_scale + fc_b[j];
        if (sum > max_score) {
            max_score = sum;
            best_idx = j;
        }
    }
    return best_idx;
}

void kai_display() {
    FRONT_COLOR=BLACK;
    LCD_ShowString(10,10,tftlcd_data.width,tftlcd_data.height,16,"Touch Test EMNIST");
}

void display_init() {
    FRONT_COLOR=RED;
    LCD_ShowString(10, 0, tftlcd_data.width, tftlcd_data.height, 16, "Char: ");
    LCD_ShowString(tftlcd_data.width - 55, 25, tftlcd_data.width, tftlcd_data.height, 16, "CLEAR");
    // [Draw color selectors...]
    LCD_Fill(120, tftlcd_data.height - 16, 139, tftlcd_data.height, BLUE);
    LCD_Fill(140, tftlcd_data.height - 16, 159, tftlcd_data.height, RED);
    LCD_Fill(160, tftlcd_data.height - 16, 179, tftlcd_data.height, MAGENTA);
    LCD_Fill(180, tftlcd_data.height - 16, 199, tftlcd_data.height, GREEN);
    LCD_Fill(200, tftlcd_data.height - 16, 219, tftlcd_data.height, CYAN);
    LCD_Fill(220, tftlcd_data.height - 16, 239, tftlcd_data.height, YELLOW);
}

int main()
{
    u8 key;
    u16 penColor = BLUE;
    u16 pcnt = 0, tcnt = 0;
    char sbuf[2]; // Can only display 1 char

    HAL_Init();
    SystemClock_Init(8,336,2,7);
    SysTick_Init(168);
    USART1_Init(115200);
    LED_Init();
    TFTLCD_Init();
    KEY_Init();
    TP_Init();

    kai_display();
    delay_ms(2000);
    LCD_Clear(WHITE);
    display_init();

    while(1)
    {
        key=KEY_Scan(0);
        if(key==KEY_UP_PRESS)
        {
            TP_Adjust();
            LCD_Clear(WHITE);
            display_init();
            pcnt = 0; tcnt = 0;
        }

        if(TP_Scan(0))
        {
            u16 px = tp_dev.x[0];
            u16 py = tp_dev.y[0];
            if (px < 5 || py < 5 || px >= tftlcd_data.width - 5 || py >= tftlcd_data.height - 5) {}
            else
            {
                if (tcnt > 3) {
                    if (pcnt > 0 && pcnt < 599 && ncr_input_buf[pcnt - 1].x != 0xFFFF) {
                        ncr_input_buf[pcnt].x = 0xFFFF;
                        ncr_input_buf[pcnt].y = 0xFFFF;
                        pcnt++;
                    }
                }
                tcnt = 0;
                if ((px > tftlcd_data.width - 60) && (py < 60))
                {
                    LCD_Fill(0, 0, tftlcd_data.width-1, tftlcd_data.height-1, WHITE);
                    display_init();
                    pcnt = 0;
                }
                else if(py > tftlcd_data.height - 18 && py < tftlcd_data.height)
                {
                    if(px>220) penColor = YELLOW;
                    else if(px>200) penColor = CYAN;
                    else if(px>180) penColor = GREEN;
                    else if(px>160) penColor = MAGENTA;
                    else if(px>140) penColor = RED;
                    else if(px>120) penColor = BLUE;
                }
                else
                {
                    LCD_Fill(px-1, py-1, px+2, py+2, penColor);
                    if (pcnt < 600)
                    {
                        if (pcnt == 0 || ncr_input_buf[pcnt - 1].x == 0xFFFF ||
                           (ncr_input_buf[pcnt - 1].x != px || ncr_input_buf[pcnt - 1].y != py))
                        {
                            ncr_input_buf[pcnt].x = px;
                            ncr_input_buf[pcnt].y = py;
                            pcnt++;
                        }
                    }
                }
            }
            delay_ms(2);
        }
        else
        {
            tcnt++;
            delay_ms(10);
            if (tcnt == 80)
            {
                if (pcnt > 1)
                {
                    static float nn_input[784];
                    preprocess_trajectory(ncr_input_buf, pcnt, nn_input);
                    u32 t_start = HAL_GetTick();
                    int predict_idx = simple_cnn_predict(nn_input);
                    u32 t_end = HAL_GetTick();

                    if (predict_idx >= 0 && predict_idx < 47) {
                        sbuf[0] = EMNIST_MAP[predict_idx];
                        sbuf[1] = '\0';
                        printf("Predicted index: %d, Char: %c, Time: %lums\r\n", predict_idx, sbuf[0], t_end - t_start);
                        LCD_Fill(74, 0, 110, 16, WHITE);
                        FRONT_COLOR = RED;
                        LCD_ShowString(74, 0, tftlcd_data.width, tftlcd_data.height, 16, (u8*)sbuf);

                        char tbuf[20];
                        sprintf(tbuf, "%lums", t_end - t_start);
                        LCD_Fill(10, 20, 120, 36, WHITE);
                        LCD_ShowString(10, 20, tftlcd_data.width, tftlcd_data.height, 16, (u8*)tbuf);
                    }
                    pcnt = 0;
                }
            }
        }
    }
}
