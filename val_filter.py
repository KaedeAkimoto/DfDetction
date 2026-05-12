# simplified_version.py
import os
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

stdout = sys.stdout
miss_img_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\miss\img")
miss_lab_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\miss\lab")

def detect_mismatched_images(img_dir, label_dir, model_path: tuple[Path, Path], iou_threshold=0.5, conf_threshold=0.25):
    """
    简化版的图片不匹配检测
    
    参数:
        img_dir: 图片目录
        label_dir: 标签目录
        model_path: 模型权重路径
        iou_threshold: IoU阈值
        conf_threshold: 置信度阈值
    """
    
    model = YOLO(model_path[1], verbose=False).cuda()
    
    # 获取图片列表
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    img_files = list(img_dir.glob('*.[jp][pn]g'))  # 支持jpg, jpeg, png
    
    mismatched = []
    miss_lab = []

    print(f"开始检查 {len(img_files)} 张图片...")
    
    for img_path in tqdm(img_files):
        # 获取对应的标签文件
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"警告: 标签文件不存在: {label_path}")
            continue
        
        # 模型预测
        sys.stdout = os.devnull
        results = model.predict(str(img_path))
        sys.stdout = stdout
        # 加载图片获取尺寸
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2] # type: ignore
        
        # 解析预测结果
        preds = results[0].boxes.cpu().numpy()  # [x1, y1, x2, y2, conf, class] # type: ignore
        # 解析标签
        with open(label_path, 'r') as f:
            labels = []
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w_norm, h_norm = map(float, parts)
                    # 转换为绝对坐标
                    x_center = x * w
                    y_center = y * h
                    width = w_norm * w
                    height = h_norm * h
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    x2 = x_center + width/2
                    y2 = y_center + height/2
                    labels.append([int(cls), x1, y1, x2, y2])
        
        # 比较数量和类别
        pred_classes = [int(cls) for idx, cls in enumerate(preds.cls) if preds.conf[idx] > conf_threshold]
        true_classes = [l[0] for l in labels]
        
        if len(pred_classes) != len(true_classes):
            mismatched.append(str(img_path.name))
            results[0].save(miss_img_dir / img_path.name)
            with (miss_lab_dir / Path(f"{img_path.stem}.txt")).open('w') as f:
                text = list(
                    [
                        str(cls), 
                        str(preds[idx].xywhn[0, 0]), 
                        str(preds[idx].xywhn[0, 1]), 
                        str(preds[idx].xywhn[0, 2]), 
                        str(preds[idx].xywhn[0, 3])
                    ]
                    for idx, cls in enumerate(preds.cls)
                )
                for obj in text:
                    f.write(" ".join(obj) + "\n")
            continue
        
        # 比较类别分布
        pred_class_count = {}
        true_class_count = {}
        
        for cls in pred_classes:
            pred_class_count[cls] = pred_class_count.get(cls, 0) + 1
        
        for cls in true_classes:
            true_class_count[cls] = true_class_count.get(cls, 0) + 1
        
        if pred_class_count != true_class_count:
            mismatched.append(str(img_path.name))
            results[0].save(miss_img_dir / img_path.name)
            with (miss_lab_dir / Path(f"{img_path.stem}.txt")).open('w') as f:
                text = list(
                    [
                        str(cls), 
                        str(preds[idx].xywhn[0, 0]), 
                        str(preds[idx].xywhn[0, 1]), 
                        str(preds[idx].xywhn[0, 2]), 
                        str(preds[idx].xywhn[0, 3])
                    ]
                    for idx, cls in enumerate(preds.cls)
                )
                for obj in text:
                    f.write(" ".join(obj) + "\n")

    
    
    # 保存结果
    output_file = "mismatched_simple.txt"
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write(f"不匹配的图片 (共{len(mismatched)}张):\n")
        f.write("="*50 + "\n")
        for img_name in mismatched:
            f.write(f"{img_name}\n")
    
    print(f"\n完成！发现 {len(mismatched)} 张不匹配的图片")
    print(f"结果保存在: {output_file}")
    
    return mismatched

# 使用示例
if __name__ == "__main__":
    # 设置你的路径
    pre_path = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\val")  # 当前目录
    img_dir = pre_path / "images"  # 图片路径
    label_dir = pre_path / "labels"  # 标签路径
    model_path = (
        Path("yolo11_c_cbam.yaml"), # 模型结构
        Path("yolo11n_c_cbam.pt")  # 模型权重文件
    )  
    
    # 运行检测
    mismatched_images = detect_mismatched_images(img_dir, label_dir, model_path)
    
    # 打印不匹配的图片
    if mismatched_images:
        print("\n不匹配的图片列表:")
        for img in mismatched_images:
            print(f"  - {img}")