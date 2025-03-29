import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision.models.resnet import ResNet, Bottleneck
from torch.utils.data import Dataset, DataLoader
from torch.hub import load_state_dict_from_url
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


class EyeDataset(Dataset):
    """加载处理好的6通道PT文件和对应的Excel标签"""

    def __init__(self, data_dir, excel_path, split='train', test_size=0.2):
        """
        Args:
            data_dir: 存放processed_*.pt文件的目录
            excel_path: 包含标签的Excel文件路径
            split: 'train'或'val'
        """
        self.data_dir = data_dir
        self.df = pd.read_excel(excel_path)

        # 分割训练集和验证集
        train_df, val_df = train_test_split(self.df, test_size=test_size, random_state=42)
        self.df = train_df if split == 'train' else val_df

        # 提取标签列 (N, D, G, C, A, H, M, O)
        self.label_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取文件名 (如 "0_left.jpg" -> "processed_0.jpg.pt")
        left_name = self.df.iloc[idx]['Left-Fundus']
        base_name = left_name.split('_')[0]  # 提取数字部分
        pt_file = f"processed_{base_name}.jpg.pt"

        # 加载6通道张量
        tensor = torch.load(os.path.join(self.data_dir, pt_file))

        # 获取标签 (转换为float32张量)
        labels = self.df.iloc[idx][self.label_columns].values.astype(np.float32)

        return tensor, torch.tensor(labels)


class MultiLabelMetrics:
    """多标签分类指标计算器"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.all_preds = []
        self.all_targets = []

    def update(self, outputs, targets):
        """累积批次预测结果"""
        preds = (torch.sigmoid(outputs) > self.threshold).int()
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

    def compute(self):
        """计算所有指标"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # 样本级准确率（完全匹配才算正确）
        accuracy = accuracy_score(targets, preds)

        # 宏平均精确度和召回率（按标签计算后平均）
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)

        return accuracy, precision, recall

    def reset(self):
        """重置累积器"""
        self.all_preds = []
        self.all_targets = []


class MultiLabelResNet(ResNet):
    """支持多通道输入和多标签分类的ResNet变体"""

    def __init__(self, block, layers, num_classes=8, in_channels=6, pretrained=False):
        super(MultiLabelResNet, self).__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if pretrained:
            self._load_pretrained_weights(in_channels)

    def _load_pretrained_weights(self, in_channels):
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet50-0676ba61.pth',
            progress=True
        )

        if in_channels != 3:
            conv1_weight = state_dict['conv1.weight']
            if in_channels == 1:
                new_conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
            else:
                new_conv1_weight = conv1_weight.repeat(1, in_channels // 3, 1, 1)
                if in_channels % 3 != 0:
                    extra_weight = torch.randn(64, in_channels % 3, 7, 7)
                    new_conv1_weight = torch.cat([new_conv1_weight, extra_weight], dim=1)
            state_dict['conv1.weight'] = new_conv1_weight

        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50_multilabel(num_classes=8, in_channels=6, pretrained=False):
    return MultiLabelResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained
    )


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_metrics = MultiLabelMetrics()

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_metrics.update(outputs, labels)

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_metrics = MultiLabelMetrics()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_running_loss += criterion(outputs, labels).item() * images.size(0)
                val_metrics.update(outputs, labels)

        # 计算指标
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        train_acc, train_prec, train_rec = train_metrics.compute()
        val_acc, val_prec, val_rec = val_metrics.compute()

        # 记录指标
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印结果
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(
            f"  Train - Loss: {epoch_train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        print(
            f"  Val   - Loss: {epoch_val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

    # 训练完成后绘制曲线
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    return model


if __name__ == "__main__":
    # 参数设置
    data_dir = "C:/Users/26448/Desktop/A07_Data/A07_Data/Training_Dataset/processed"  # 替换为你的PT文件目录
    excel_path = "C:/Users/26448/Desktop/lable.xlsx"  # 替换为你的Excel标签路径
    batch_size = 8
    epochs = 10
    lr = 1e-4

    # 创建数据集和数据加载器
    train_dataset = EyeDataset(data_dir, excel_path, split='train')
    val_dataset = EyeDataset(data_dir, excel_path, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    model = resnet50_multilabel(num_classes=8, in_channels=6, pretrained=True)

    # 打印模型结构
    print(model)

    # 训练验证
    trained_model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

    # 保存最终模型
    torch.save(trained_model.state_dict(), 'final_model.pth')
    print("Training completed and models saved!")