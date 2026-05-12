from HyperParameters import HyperParameters
from ultralytics import YOLO
import multiprocessing
from collections.abc import Iterable
from argparse import Namespace


def single_train(train_model: Namespace):
    model = YOLO(train_model.model)
    model.load("yolo11n.pt")
    model.train(
        **train_model.__dict__,
    )
    model.val()


def train_loop(train_models: Iterable[Namespace]):
    for train_model in train_models:
        single_train(train_model)


def increment_single_train(train_model: Namespace, epochs: int):
    backup = train_model.epochs
    train_model.epochs = epochs
    resume_weight = train_model.project + "/train3/weights/last.pt"

    model = YOLO(resume_weight)
    model.train(
        **train_model.__dict__,
    )
    train_model.epochs = backup
    model.val()

def main():
    multiprocessing.freeze_support()
    # train_loop([
    #     HyperParameters._1v0_RUN,
    #     HyperParameters._1v1_RUN,
    #     HyperParameters._2v0_RUN,
    # ])
    increment_single_train(HyperParameters._2v0_RUN, 100)


if __name__ == '__main__':
    main()
