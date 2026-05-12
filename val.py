import torch
from ultralytics import YOLO
import multiprocessing


torch.use_deterministic_algorithms(False)


if __name__ == '__main__':
    # 设置多进程启动方法
    multiprocessing.freeze_support()
    
    # 加载模型
    model = YOLO('yolo11_c_cbam.yaml')
    model.load("yolo11n_c_cbam.pt")
    

    results = model.val(
        data='DefectDetection.yaml', 
        epochs=100,
        imgsz=640, 
        batch=32,
        device=0,
        workers=0,
        name='yolo11n_c_cbam_val',
        conf=0.4,  # 置信度阈值
        iou=0.6,    # IoU阈值
    )
    
    if hasattr(results, 'results_dict'):
            print("验证结果:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value:.4f}")