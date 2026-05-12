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
        iou=0.7, 
        project='runs/train/_2v0_RUN',
        **TrainHyperParameters.FullRounds.__dict__,
    ),
    _3v0_RUN=Namespace (
        # ...
    ),
)