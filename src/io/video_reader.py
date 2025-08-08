import cv2

def open_video(source):
    """OpenCV VideoCapture wrapper for path/int/RTSP."""
    return cv2.VideoCapture(source)

def read_and_resize(cap, width, height):
    ret, frame = cap.read()
    if not ret:
        return False, None
    if width and height:
        frame = cv2.resize(frame, (width, height))
    return True, frame
