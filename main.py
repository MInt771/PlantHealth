import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import argparse


# ================== 配置部分 ==================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='植物病害严重度分类数据集构建工具')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='PlantVillage数据集根目录路径（包含各个作物文件夹）')
    parser.add_argument('--dst_dir', type=str, default='./dataset',
                        help='输出数据集路径（默认./dataset）')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式（显示病害区域可视化）')
    return parser.parse_args()


# ================== 核心功能 ==================
def calculate_disease_area(img_path, debug=False):
    """计算病害区域占比（支持调试可视化）
    Args:
        img_path: 图片路径
        debug: 是否显示病害区域可视化
    Returns:
        float: 病害区域占比（0.0~1.0）
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片 {img_path}")
        return 0.0

    # HSV颜色空间转换
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义病害颜色范围（黄/褐/黑）
    lower_yellow = np.array([20, 50, 50])  # 黄色病害
    upper_yellow = np.array([30, 255, 255])
    lower_dark = np.array([0, 50, 0])  # 深色病害
    upper_dark = np.array([180, 255, 100])

    # 创建颜色掩膜
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_dark)

    # 调试可视化
    if debug:
        vis = cv2.bitwise_and(img, img, mask=combined_mask)
        cv2.imshow('Disease Area', np.hstack([img, vis]))
        cv2.waitKey(100)  # 显示100ms

    # 计算病害区域占比
    disease_ratio = np.sum(combined_mask > 0) / (img.shape[0] * img.shape[1])
    return disease_ratio


def build_dataset(args):
    """构建分类数据集"""
    # 目标文件夹结构
    CLASSES = ['healthy', 'mild', 'severe']
    for cls in CLASSES:
        os.makedirs(os.path.join(args.dst_dir, cls), exist_ok=True)

    # 你提供的文件夹列表
    folders = [
        "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
        "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
        "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato_Bacterial_spot",
        "Tomato_Early_blight", "Tomato_healthy", "Tomato_Late_blight",
        "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite"
    ]

    # 遍历所有文件夹
    for folder in tqdm(folders, desc='处理作物类别'):
        src_dir = os.path.join(args.src_dir, folder)
        if not os.path.exists(src_dir):
            print(f"警告：文件夹不存在 {src_dir}")
            continue

        # 健康样本直接复制
        if "healthy" in folder.lower():
            for file in tqdm(os.listdir(src_dir), desc=f'复制健康样本 {folder}', leave=False):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(
                        os.path.join(src_dir, file),
                        os.path.join(args.dst_dir, 'healthy', f"{folder}_{file}")
                    )
        # 病害样本计算严重程度
        else:
            for file in tqdm(os.listdir(src_dir), desc=f'处理病害样本 {folder}', leave=False):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(src_dir, file)
                    ratio = calculate_disease_area(img_path, args.debug)

                    if ratio >= 0.5:
                        dest = 'severe'
                    elif ratio >= 0.1:
                        dest = 'mild'
                    else:  # 小于10%的视为健康（可能存在误标）
                        dest = 'healthy'

                    shutil.copy(
                        img_path,
                        os.path.join(args.dst_dir, dest, f"{folder}_{file}")
                    )


# ================== 主程序 ==================
if __name__ == '__main__':
    args = parse_args()

    # 验证源路径
    if not os.path.exists(args.src_dir):
        raise FileNotFoundError(f"源目录不存在: {args.src_dir}")

    # 检查示例文件夹
    sample_folder = os.path.join(args.src_dir, "Tomato_healthy")
    if not os.path.exists(sample_folder):
        print(f"警告：示例文件夹不存在 {sample_folder}，请确认路径是否正确")

    print(f"开始构建数据集，源目录: {args.src_dir}")
    build_dataset(args)
    print(f"数据集构建完成！输出目录: {args.dst_dir}")

    # 统计各类别数量
    print("\n类别分布统计:")
    for cls in ['healthy', 'mild', 'severe']:
        count = len(os.listdir(os.path.join(args.dst_dir, cls)))
        print(f"{cls}: {count} 张图片")