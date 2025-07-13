import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import argparse

CLASSES = ['healthy', 'mild', 'severe']  # 全局类别定义
# ================== 配置部分 ==================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='植物病害严重度分类数据集构建工具-精准版')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='PlantVillage数据集根目录路径')
    parser.add_argument('--dst_dir', type=str, default='./dataset',
                        help='输出数据集路径（默认./dataset）')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式（显示病害检测过程）')
    parser.add_argument('--recheck', action='store_true',
                        help='启用自动校验模式（检测healthy文件夹误判）')
    return parser.parse_args()


# ================== 核心功能 ==================
def enhanced_disease_detection(img_path, debug=False):
    """升级版病害检测：颜色+纹理+形态学特征融合"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片 {img_path}")
        return 0.0

    # 1. 颜色特征（HSV空间）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 40, 40])  # 调整颜色范围
    upper_yellow = np.array([35, 255, 255])
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([25, 255, 255])
    mask_color = cv2.bitwise_or(
        cv2.inRange(hsv, lower_yellow, upper_yellow),
        cv2.inRange(hsv, lower_brown, upper_brown)
    )

    # 2. 纹理特征（LBP算法简化版）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = cv2.Laplacian(gray, cv2.CV_64F)
    _, texture_mask = cv2.threshold(np.uint8(np.absolute(lbp)), 30, 255, cv2.THRESH_BINARY)

    # 3. 形态学特征（边缘检测）
    edges = cv2.Canny(gray, 50, 150)

    # 特征融合（加权计算）
    combined_mask = cv2.bitwise_or(
        mask_color,
        cv2.bitwise_and(texture_mask, edges)
    )

    # 后处理（去除小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 计算病害区域占比
    disease_ratio = np.sum(processed_mask > 0) / (img.shape[0] * img.shape[1])

    # 调试可视化
    if debug:
        vis = cv2.bitwise_and(img, img, mask=processed_mask)
        cv2.putText(vis, f"Ratio: {disease_ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Debug', np.hstack([img, vis]))
        cv2.waitKey(100 if not args.debug else 0)

    return disease_ratio


def dynamic_classify(ratio):
    """动态阈值分类（基于统计分布调整）"""
    if ratio < 0.01:  # 健康阈值
        return "healthy"
    elif ratio < 0.4:  # 一般病害阈值
        return "mild"
    else:  # 严重病害
        return "severe"


def recheck_healthy_folder(folder_path):
    """自动校验healthy文件夹中的误分类"""
    recheck_count = 0
    for img_name in tqdm(os.listdir(folder_path), desc="校验healthy文件夹"):
        img_path = os.path.join(folder_path, img_name)
        ratio = enhanced_disease_detection(img_path)

        if ratio > 0.15:  # 高于阈值则重新分类
            dest = dynamic_classify(ratio)
            shutil.move(img_path, os.path.join(args.dst_dir, dest, img_name))
            recheck_count += 1
    return recheck_count


# ================== 数据集构建 ==================
def build_dataset(args):
    """构建分类数据集（严格两阶段处理）"""
    # 创建目标文件夹
    CLASSES = ['healthy', 'mild', 'severe']
    for cls in CLASSES:
        os.makedirs(os.path.join(args.dst_dir, cls), exist_ok=True)

    # 获取所有文件夹（自动识别带healthy的）
    all_folders = [f for f in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, f))]

    # ===== 第一阶段：处理healthy文件夹 =====
    print("=== 第一阶段：提取健康样本 ===")
    healthy_folders = [f for f in all_folders if "healthy" in f.lower()]
    for folder in tqdm(healthy_folders, desc="复制健康样本"):
        src_dir = os.path.join(args.src_dir, folder)
        for file in os.listdir(src_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(
                    os.path.join(src_dir, file),
                    os.path.join(args.dst_dir, 'healthy', f"{folder}_{file}")
                )

    # ===== 第二阶段：处理病害文件夹 =====
    print("\n=== 第二阶段：分类病害样本 ===")
    disease_folders = [f for f in all_folders if "healthy" not in f.lower()]
    for folder in tqdm(disease_folders, desc="分类病害样本"):
        src_dir = os.path.join(args.src_dir, folder)
        for file in os.listdir(src_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(src_dir, file)
                ratio = enhanced_disease_detection(img_path, args.debug)
                dest = dynamic_classify(ratio)
                shutil.copy(
                    img_path,
                    os.path.join(args.dst_dir, dest, f"{folder}_{file}")
                )


# ================== 主程序 ==================
if __name__ == '__main__':
    args = parse_args()

    # 验证路径
    if not os.path.exists(args.src_dir):
        raise FileNotFoundError(f"源目录不存在: {args.src_dir}")

    # 构建数据集
    print(f"开始构建数据集...")
    build_dataset(args)

    # 自动校验（可选）
    if args.recheck:
        healthy_dir = os.path.join(args.dst_dir, 'healthy')
        if os.path.exists(healthy_dir):
            print("\n=== 校验阶段：复查健康样本 ===")
            corrected = recheck_healthy_folder(healthy_dir)
            print(f"校验完成，共修正 {corrected} 个误分类样本")

    # 统计结果
    print("\n=== 最终类别分布 ===")
    for cls in CLASSES:
        count = len(os.listdir(os.path.join(args.dst_dir, cls)))
        print(f"{cls}: {count} 张图片")