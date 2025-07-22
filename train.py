# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# import torch.nn as nn
# import torch.optim as optim
# import os
# from PIL import Image
# from tqdm import tqdm
#
#
# # ================== 配置部分 ==================
# class Config:
#     data_dir = "./dataset"  # 修改为你的实际路径（包含healthy/mild/severe）
#     classes = ["healthy", "mild", "severe"]
#     batch_size = 32
#     num_workers = 4
#     lr = 0.001
#     epochs = 50
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ================== 数据集类 ==================
# class PlantDiseaseDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data = []
#         self.transform = transform or self.default_transform()
#
#         # 直接读取my_dataset下的分类文件夹
#         for label_idx, class_name in enumerate(Config.classes):
#             class_dir = os.path.join(data_dir, class_name)
#             if not os.path.exists(class_dir):
#                 raise FileNotFoundError(f"目录不存在: {class_dir}")
#
#             for img_name in os.listdir(class_dir):
#                 self.data.append((os.path.join(class_dir, img_name), label_idx))
#
#     def default_transform(self):
#         return transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         img_path, label = self.data[idx]
#         img = Image.open(img_path).convert("RGB")
#         return self.transform(img), torch.tensor(label)
#
#
# # ================== 模型训练 ==================
# def train_model():
#     # 1. 检查GPU是否可用
#     print(f"当前设备: {Config.device}")
#     if Config.device.type == "cpu":
#         print("警告: 未检测到GPU，将使用CPU训练（速度较慢）")
#
#     # 2. 准备数据（不再区分train/val）
#     full_dataset = PlantDiseaseDataset(Config.data_dir)
#
#     # 手动划分训练集和验证集（8:2）
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
#
#     # 3. 初始化模型
#     model = models.efficientnet_b0(pretrained=True)
#     model.classifier[1] = nn.Linear(1280, len(Config.classes))
#     model = model.to(Config.device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=Config.lr)
#
#     # 4. 训练循环
#     for epoch in range(Config.epochs):
#         model.train()
#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}"):
#             inputs, labels = inputs.to(Config.device), labels.to(Config.device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#         # 验证
#         val_acc = evaluate(model, val_loader)
#         print(f"Val Acc: {val_acc:.2f}%")
#
#     torch.save(model.state_dict(), "plant_disease_model.pth")
#
#
# def evaluate(model, dataloader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(Config.device), labels.to(Config.device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return 100 * correct / total
#
#
# if __name__ == "__main__":
#     # 检查数据路径
#     if not os.path.exists(Config.data_dir):
#         raise FileNotFoundError(f"数据目录不存在: {Config.data_dir}")
#
#     # 检查GPU是否可用
#     if torch.cuda.is_available():
#         print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         print("未检测到GPU，将使用CPU训练")
#
#     train_model()






#上方为旧代码
"""上方为旧代码"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ================== 配置部分 ==================
class Config:
    data_dir = "./dataset"  # 修改为你的实际路径（包含healthy/mild/severe）
    classes = ["healthy", "mild", "severe"]
    batch_size = 32
    num_workers = 4
    lr = 0.0005  # 降低学习率
    epochs = 30  # 减少epochs但增加早停
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stop_patience = 5  # 早停耐心值
    weight_decay = 1e-4  # L2正则化


# ================== 数据集类 ==================
class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform or self.default_transform()

        for label_idx, class_name in enumerate(Config.classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"目录不存在: {class_dir}")

            for img_name in os.listdir(class_dir):
                self.data.append((os.path.join(class_dir, img_name), label_idx))

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


# ================== 训练增强 ==================
def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def create_data_loaders():
    full_dataset = PlantDiseaseDataset(Config.data_dir)

    # 计算类别权重
    class_counts = np.zeros(len(Config.classes))
    for _, label in full_dataset.data:
        class_counts[label] += 1

    print("\n类别分布:")
    for i, count in enumerate(class_counts):
        print(f"{Config.classes[i]}: {count} samples")

    # 8:1:1 划分训练集、验证集和测试集
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    # 创建数据集
    train_dataset = PlantDiseaseDataset(Config.data_dir, transform=get_train_transform())
    val_dataset = PlantDiseaseDataset(Config.data_dir)
    test_dataset = PlantDiseaseDataset(Config.data_dir)

    # 计算采样权重
    class_weights = 1. / class_counts
    sample_weights = torch.tensor([class_weights[label] for _, label in full_dataset.data])

    # 只对训练集使用加权采样
    train_sampler = WeightedRandomSampler(
        weights=[sample_weights[i] for i in train_indices.indices],
        num_samples=len(train_indices),
        replacement=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size,
        sampler=train_sampler, num_workers=Config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=Config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=Config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_counts


def initialize_model(class_counts):
    # 使用预训练的EfficientNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 修改分类头
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(Config.classes))

    # 计算类别权重用于损失函数
    class_weights = torch.tensor(class_counts.sum() / class_counts, dtype=torch.float32)
    class_weights = class_weights.to(Config.device)

    # 使用加权交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 使用AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )

    # 学习率调度器（移除了verbose参数）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2
    )

    model = model.to(Config.device)
    return model, criterion, optimizer, scheduler


def train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader):
    best_acc = 0.0
    best_model = None
    patience_counter = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(Config.epochs):
        model.train()
        running_loss = 0.0

        # 训练阶段
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        val_acc, val_report, val_cm = evaluate(model, val_loader)
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch + 1}/{Config.epochs}:")
        print(f"Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("Classification Report:")
        print(val_report)

        # 更新学习率
        scheduler.step(val_acc)

        # 早停和保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pth")
            patience_counter = 0
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= Config.early_stop_patience:
                print(f"Early stopping after {Config.early_stop_patience} epochs without improvement")
                break

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()

    return best_model


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    # 生成分类报告和混淆矩阵
    report = classification_report(
        all_labels, all_preds,
        target_names=Config.classes,
        digits=4
    )

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, Config.classes)

    return accuracy, report, cm


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def test_model(model, test_loader):
    model.eval()
    test_acc, test_report, _ = evaluate(model, test_loader)
    print("\nFinal Test Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("Classification Report:")
    print(test_report)


def main():
    print(f"Using device: {Config.device}")
    if Config.device.type == "cpu":
        print("Warning: Training on CPU will be slow!")

    # 准备数据
    train_loader, val_loader, test_loader, class_counts = create_data_loaders()

    # 初始化模型
    model, criterion, optimizer, scheduler = initialize_model(class_counts)

    # 训练和验证
    best_model = train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader)

    # 加载最佳模型并测试
    model.load_state_dict(best_model)
    test_model(model, test_loader)

    print("\nTraining completed. Best model saved as 'best_model.pth'")


if __name__ == "__main__":
    main()