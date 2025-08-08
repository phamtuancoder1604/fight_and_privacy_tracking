import cv2
import numpy as np

# Tuỳ chọn: bản sao hàm để dùng lại ở nơi khác mà không động chạm file gốc
def blur_or_mosaic_face(image, face_boxes=None, face_landmarks=None, method="blur", blur_level=25, mosaic_size=15):
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
