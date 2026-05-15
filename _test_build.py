import sys
sys.path.insert(0, 'ultralytics-main')
from ultralytics import YOLO
import torch

for name, yaml_file in [
    ('10v (PConv+C2fLSK)', 'yolo11_fix_10.yaml'),
    ('11v (DCNv3+AKConv)', 'yolo11_fix_11.yaml'),
    ('12v (A2+DyHeadv3)', 'yolo11_fix_12.yaml'),
    ('13v (RELAN+CSDN)', 'yolo11_fix_13.yaml'),
]:
    print(f'\n=== Building {name} ({yaml_file}) ===')
    try:
        model = YOLO(yaml_file)
        x = torch.randn(1, 3, 640, 640)
        out = model.model(x)
        if isinstance(out, (list, tuple)):
            print(f'  Output shapes: {[o.shape for o in out]}')
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f'  Parameters: {n_params/1e6:.2f}M')
        print(f'  BUILD SUCCESS!')
    except Exception as e:
        print(f'  BUILD FAILED: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        print(f'  --- END OF ERROR ---')