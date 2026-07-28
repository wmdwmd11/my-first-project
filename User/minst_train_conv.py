import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 1. Define the Lightweight CNN Architecture
class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0)  # Output: 8x26x26
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 8x13x13
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0)  # Output: 16x11x11
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 16x5x5
        self.fc = nn.Linear(16 * 5 * 5, 10)  # Output: 10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = self.fc(x)
        return x


# 2. Define Custom Dataset for CSV files
class MNISTCsvDataset(Dataset):
    def __init__(self, csv_file):
        # 读取 CSV 数据
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 第 0 列是 label
        label = int(self.data_frame.iloc[idx, 0])

        # 第 1 列及之后是 784 个像素特征
        pixels = self.data_frame.iloc[idx, 1:].values.astype(np.float32)

        # 将一维的 784 转换为 1x28x28 的张量，并归一化到 0.0~1.0 范围
        pixels = pixels.reshape(1, 28, 28) / 255.0
        image = torch.tensor(pixels)

        return image, label


# 3. Training Function
def train_model():
    print("Loading local CSV train dataset...")
    train_dataset = MNISTCsvDataset(csv_file='./minst/mnist_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = LightCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}")
                running_loss = 0.0

    print("Training finished!\n")
    return model, device


# 4. Evaluation Function
def evaluate_model(model, device):
    print("Loading local CSV test dataset...")
    test_dataset = MNISTCsvDataset(csv_file='./minst/mnist_test.csv')
    # 测试集不需要打乱 (shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 切换到评估模式，关闭 Dropout 和 BatchNorm 等训练期特有行为
    model.eval()
    correct = 0
    total = 0

    print("Starting evaluation...")
    # 关闭梯度计算以节省显存并加速前向传播
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 找到概率最大的一项作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy on {total} test images: {accuracy:.2f}%\n")
    return accuracy


# 5. Helper function to export Tensors to C arrays
def export_tensor_to_c(tensor, name, file):
    flat = tensor.detach().cpu().numpy().flatten()
    file.write(f"const float {name}[{len(flat)}] = {{\n")
    # Format to 6 decimal places, 8 numbers per line
    lines = [", ".join([f"{x:.6f}f" for x in flat[i:i + 8]]) for i in range(0, len(flat), 8)]
    file.write(",\n".join(lines))
    file.write("\n};\n\n")


# 6. Generate weights.h
def export_weights(model):
    print("Exporting weights to conv_weights.h...")
    with open("conv_weights.h", "w") as f:
        f.write("#ifndef __WEIGHTS_H\n#define __WEIGHTS_H\n\n")

        export_tensor_to_c(model.conv1.weight, "conv1_w", f)
        export_tensor_to_c(model.conv1.bias, "conv1_b", f)

        export_tensor_to_c(model.conv2.weight, "conv2_w", f)
        export_tensor_to_c(model.conv2.bias, "conv2_b", f)

        # Transpose the Fully Connected weights!
        export_tensor_to_c(model.fc.weight.T, "fc_w", f)
        export_tensor_to_c(model.fc.bias, "fc_b", f)

        f.write("#endif\n")
    print("conv_weights.h generation complete!")


if __name__ == "__main__":
    # 1. 训练模型，并获取模型和设备信息
    trained_model, compute_device = train_model()

    # 2. 在测试集上验证准确率
    evaluate_model(trained_model, compute_device)

    # 3. 导出权重
    export_weights(trained_model)