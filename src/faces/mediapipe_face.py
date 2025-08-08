import mediapipe as mp

class MPFaceDetector:
    def __init__(self, model_selection=0, min_detection_confidence=0.4):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=model_selection,
                                                   min_detection_confidence=min_detection_confidence)

    def process(self, rgb_image):
        return self.detector.process(rgb_image)
