from argparse import Namespace


BaseModelParameters = Namespace (
        imgsz=640, 
        device=0,
        workers=0,
        data='DefectDetection.yaml',
)


TrainHyperParameters = Namespace (
    FewRounds=Namespace(
        epochs=100,
        **BaseModelParameters.__dict__,
    ),
    FullRounds=Namespace(
        epochs=300,
        **BaseModelParameters.__dict__,
    ),
    ExtendedRounds=Namespace(
        epochs=500,
        **BaseModelParameters.__dict__,
    ),
)



HyperParameters = Namespace (
    _0v0_RUN = Namespace(
        model='yolo11.yaml',
        batch=96,
        project='runs/train/_0v0_RUN',
        **TrainHyperParameters.FullRounds.__dict__,
    ),
    _1v0_RUN = Namespace(
        model='yolo11_fix_1.yaml',
        batch=56,
        iou=0.7, 
        project='runs/train/_1v0_RUN',
        **TrainHyperParameters.FullRounds.__dict__,
    ),
    _1v1_RUN = Namespace(
        model='yolo11_fix_1.yaml',
        batch=56,
        project='runs/train/_1v1_RUN',
        **TrainHyperParameters.FullRounds.__dict__,
    ),
    _2v0_RUN=Namespace (
        model='yolo11_fix_2.yaml',
        batch=32,
        project='runs/train/_2v0_RUN',
        **TrainHyperParameters.ExtendedRounds.__dict__,
    ),
    _3v0_RUN=Namespace (
        model='yolo11_fix_3.yaml',
        batch=36,
        cos_lr=True,
        warmup_epochs=5,
        project='runs/train/_3v0_RUN',
        **TrainHyperParameters.ExtendedRounds.__dict__,
    ),
    _4v0_RUN=Namespace (
        model='yolo11_fix_4.yaml',
        batch=32,
        iou=0.7,
        cos_lr=True,
        warmup_epochs=5,
        label_smoothing=0.1,
        weight_decay=0.001,
        dropout=0.1,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        mixup=0.1,
        copy_paste=0.1,
        project='runs/train/_4v0_RUN',
        **TrainHyperParameters.ExtendedRounds.__dict__,
    ),
)