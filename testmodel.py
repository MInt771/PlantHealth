import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import sys


# ===== 配置部分 =====
class Config:
    model_path = "plant_disease_model.pth"  # 你的训练好的模型权重
    classes = ["healthy", "mild", "severe"]  # 必须和训练时一致
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 本地模型路径（指向解压后的文件夹）
    local_repo = "hub/NVIDIA-DeepLearningExamples-a6c678e/NVIDIA-DeepLearningExamples-a6c678e"


# ===== 1. 加载模型（完全本地化） =====
def load_model():
    # 添加本地模型路径到Python路径
    sys.path.insert(0, Config.local_repo)

    # 从本地加载模型
    model = torch.hub.load(
        Config.local_repo,
        'nvidia_efficientnet_b0',
        pretrained=True,
        source='local'
    )

    # 修改最后一层（三分类）
    model.classifier.fc = torch.nn.Linear(1280, len(Config.classes))

    # 加载自定义权重（带自动修复）
    if os.path.exists(Config.model_path):
        state_dict = torch.load(Config.model_path, map_location=Config.device)

        # 权重键名修复（处理新旧版本差异）
        fixed_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("features.8"):  # 处理特征层命名
                new_key = k.replace("features.8", "features")
            elif k.startswith("classifier.1"):  # 处理分类层命名
                new_key = k.replace("classifier.1", "classifier.fc")
            fixed_dict[new_key] = v

        # 加载修复后的权重
        model.load_state_dict(fixed_dict, strict=False)
        print(f"成功加载自定义权重: {Config.model_path}")
    else:
        print(f"警告: 未找到权重文件 {Config.model_path}, 使用预训练权重")

    model = model.to(Config.device)
    model.eval()
    return model


# ===== 2. 图像预处理 =====
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(Config.device)


# ===== 3. 预测函数 =====
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_prob, pred_class = torch.max(probs, 1)
    return pred_class.item(), pred_prob.item()


# ===== 主程序 =====
if __name__ == "__main__":
    # 初始化模型
    print("正在从本地加载模型...")
    model = load_model()
    print(f"模型加载完成 | 设备: {Config.device} | 类别: {Config.classes}")


    # 单图预测
    def predict_single(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = preprocess_image(img)
            pred_class, pred_prob = predict(model, img_tensor)
            print(f"\n预测结果: {Config.classes[pred_class]} (置信度: {pred_prob:.2%})")

            # 显示带标注的图像
            img_np = np.array(img)
            cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            label = f"{Config.classes[pred_class]} ({pred_prob:.1%})"
            color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)
            cv2.putText(cv_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Prediction", cv_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"错误: {str(e)}")


    # 摄像头实时检测
    def camera_predict():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("\n实时检测模式 (按Q退出)")
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 转换和预测
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess_image(img)
            pred_class, pred_prob = predict(model, img_tensor)

            # 显示结果
            label = f"{Config.classes[pred_class]} ({pred_prob:.1%})"
            color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Plant Health Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    # 批量预测
    def batch_predict(folder_path):
        if not os.path.isdir(folder_path):
            print("错误: 文件夹不存在")
            return

        print(f"\n批量预测: {folder_path}")
        results = []
        for img_name in sorted(os.listdir(folder_path)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(folder_path, img_name)
                    img = Image.open(img_path).convert("RGB")
                    pred_class, pred_prob = predict(model, preprocess_image(img))
                    result = f"{img_name:<25} -> {Config.classes[pred_class]:<10} ({pred_prob:.1%})"
                    print(result)
                    results.append(result)
                except Exception as e:
                    print(f"{img_name} 处理失败: {str(e)}")

        # 保存结果到txt文件
        with open("prediction_results.txt", "w") as f:
            f.write("\n".join(results))
        print("\n结果已保存到 prediction_results.txt")


    # 交互菜单
    while True:
        print("\n=== 植物健康检测系统 ===")
        print("1. 单张图片预测")
        print("2. 摄像头实时检测")
        print("3. 批量预测文件夹")
        print("4. 退出")
        choice = input("请选择模式 (1-4): ").strip()

        if choice == "1":
            img_path = input("输入图片路径: ").strip('"\' ')
            predict_single(img_path)
        elif choice == "2":
            camera_predict()
        elif choice == "3":
            folder = input("输入文件夹路径: ").strip('"\' ')
            batch_predict(folder)
        elif choice == "4":
            print("程序退出")
            break
        else:
            print("无效输入，请重新选择")