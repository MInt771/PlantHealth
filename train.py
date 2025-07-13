import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from tqdm import tqdm


# ================== 配置部分 ==================
class Config:
    data_dir = "./dataset"  # 修改为你的实际路径（包含healthy/mild/severe）
    classes = ["healthy", "mild", "severe"]
    batch_size = 32
    num_workers = 4
    lr = 0.001
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== 数据集类 ==================
class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform or self.default_transform()

        # 直接读取my_dataset下的分类文件夹
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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label)


# ================== 模型训练 ==================
def train_model():
    # 1. 检查GPU是否可用
    print(f"当前设备: {Config.device}")
    if Config.device.type == "cpu":
        print("警告: 未检测到GPU，将使用CPU训练（速度较慢）")

    # 2. 准备数据（不再区分train/val）
    full_dataset = PlantDiseaseDataset(Config.data_dir)

    # 手动划分训练集和验证集（8:2）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

    # 3. 初始化模型
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, len(Config.classes))
    model = model.to(Config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # 4. 训练循环
    for epoch in range(Config.epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        val_acc = evaluate(model, val_loader)
        print(f"Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "plant_disease_model.pth")


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    # 检查数据路径
    if not os.path.exists(Config.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {Config.data_dir}")

    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到GPU，将使用CPU训练")

    train_model()