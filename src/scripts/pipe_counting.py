from yolov5.detect import run
from yolov5.detect import load_model


def load_yolo_model(model_path: str):
    return load_model(weights=model_path)


def perform_infer(image_path: str,model):
    return run(source=image_path,model1=model,imgsz=(1000,1000) )
