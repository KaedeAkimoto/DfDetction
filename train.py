from HyperParameters import HyperParameters
from ultralytics import YOLO
import multiprocessing
from collections.abc import Iterable, Callable
from typing import Any, Annotated
from argparse import Namespace


yolo_result_set = Annotated[Any, "results of model.train/model.val"]
results_tuple = Annotated[tuple[yolo_result_set, yolo_result_set], "(model.train, model.val)"]


def update_control_params(model: YOLO, params: dict): 
    for k, v in params.items():
        model.model.args[k] = v

def export_model(model: YOLO, results: results_tuple, fmt: str = 'onnx') -> results_tuple:
    model.export(format=fmt)
    return results


def train(
        train_model: Namespace, 
        controls: Callable[[YOLO, dict], Any] = lambda model, params: None, 
        load_weight: bool = False,
        after_train: Callable[[YOLO, results_tuple], Any] = export_model
    ) -> results_tuple | Any:
    assert train_model.model is not None, "model is None"
    
    model = None
    if load_weight:
        assert train_model.load_weight is not None, "load_weight is None"
        model = YOLO(train_model.load_weight)
    else: 
        model = YOLO(train_model.model)

    assert model is not None, "model is None"

    train_params = train_model.__dict__
    controls(model, train_model.control_params)
    multiprocessing.freeze_support()
    train_params.pop('control_params', None)
    train_params.pop('load_weight', None)
    train_params.pop("model", None)
    print(train_params)
    train_results = model.train(
        **train_params,
    )
    val_results = model.val()
    results = (train_results, val_results)
    rtv = results
    try:
        rtv = after_train(model, results)

    except Exception as e:
        print("after_train failed")
        repr(e)

    finally:
        return rtv


def main():
    train(HyperParameters._6v0_RUN,controls=update_control_params)



if __name__ == '__main__':
    main()
