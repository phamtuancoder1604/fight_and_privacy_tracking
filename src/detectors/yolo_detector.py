from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, classes=None, conf=0.25):
        self.model = YOLO(model_path)
        self.classes = classes if classes is not None else [0]
        self.conf = conf

    def infer(self, frame):
        return self.model(frame, classes=self.classes, conf=self.conf)
