from HyperParameters import HyperParameters
from ultralytics import YOLO
import multiprocessing
from collections.abc import Iterable
from argparse import Namespace


def single_train(train_model: Namespace):
    model = YOLO(train_model.model)
    model.load("yolo11n.pt")
    model.model.args['inner_iou'] = True
    model.model.args['inner_ratio'] = 0.7
    model.train(
        **train_model.__dict__,
    )
    model.val()


def train_loop(train_models: Iterable[Namespace]):
    for train_model in train_models:
        single_train(train_model)


def increment_single_train(train_model: Namespace, epochs: int, times: int | None = None):
    backup = train_model.epochs
    train_model.epochs = epochs
    weight = "/train/weights/last.pt" if times is None else f"/train{times}/weights/best.pt" 
    resume_weight = train_model.project + weight

    model = YOLO(resume_weight)
    model.train(
        **train_model.__dict__,
    )
    train_model.epochs = backup
    model.val()


def main():
    multiprocessing.freeze_support()
    single_train(HyperParameters._5v0_RUN)


if __name__ == '__main__':
    main()
