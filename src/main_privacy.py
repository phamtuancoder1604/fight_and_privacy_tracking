import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

def blur_or_mosaic_face(
    image: np.ndarray, 
    face_boxes: list = None, 
    face_landmarks: list = None, 
    method: str = "blur", 
    blur_level: int = 25, 
    mosaic_size: int = 15
):
    img = image.copy()
    if face_boxes:
        for box in face_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            face_roi = img[y1:y2, x1:x2]
            if face_roi.size == 0: continue
            if method == "blur":
                face_roi = cv2.GaussianBlur(face_roi, (blur_level|1, blur_level|1), 0)
            elif method == "mosaic":
                h, w = face_roi.shape[:2]
                temp = cv2.resize(face_roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
                face_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y1:y2, x1:x2] = face_roi
    if face_landmarks:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for landmarks in face_landmarks:
            points = np.array(landmarks, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        if method == "blur":
            blurred = cv2.GaussianBlur(img, (blur_level|1, blur_level|1), 0)
            img = np.where(mask[:, :, None] == 255, blurred, img)
        elif method == "mosaic":
            mosaic = img.copy()
            mosaic = cv2.resize(mosaic, (img.shape[1]//mosaic_size, img.shape[0]//mosaic_size), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(mosaic, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img = np.where(mask[:, :, None] == 255, mosaic, img)
    return img


input_width, input_height = 1600, 900 


yolo_model = YOLO("models/yolov8x.pt")
tracker = DeepSort(
    max_age=150,          
    n_init=2,
    nms_max_overlap=0.7     
)

# Mediapipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.40)

video_path = "videos/walking _360.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (input_width, input_height))
    

    frame_flipped = cv2.flip(frame, 1)
    # Detect trên cả hai ảnh: gộp box phát hiện lại
    results = yolo_model(frame, classes=[0], conf=0.18)
    results_flip = yolo_model(frame_flipped, classes=[0], conf=0.18)

    detections = []
    person_boxes = []


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person', None))
            person_boxes.append([x1, y1, x2, y2])


    for result in results_flip:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1_new = input_width - x2
            x2_new = input_width - x1

            if (x2_new - x1_new) < 30 or (y2 - y1) < 30:
                continue
            conf = float(box.conf[0])
            detections.append(([x1_new, y1, x2_new-x1_new, y2-y1], conf, 'person', None))
            person_boxes.append([x1_new, y1, x2_new, y2])


    tracks = tracker.update_tracks(detections, frame=frame)


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_boxes = []
    results_face = face_detection.process(rgb)
    if results_face.detections:
        for detection in results_face.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bbox.xmin * iw)
            y1 = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            x2 = x1 + w
            y2 = y1 + h
            for pb in person_boxes:
                if x1 >= pb[0] and y1 >= pb[1] and x2 <= pb[2] and y2 <= pb[3]:
                    face_boxes.append([x1, y1, x2, y2])
                    break

    frame = blur_or_mosaic_face(frame, face_boxes=face_boxes, method="mosaic", mosaic_size=15)


    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Tracking + Face Privacy (Advanced Optimized)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()