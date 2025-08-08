import cv2
import numpy as np
import csv
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def get_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

def compute_iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def now_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')


VIDEO_PATH  = "videos/fight_2.mp4"
LOG_CSV     = "fight_events.csv"
WIDTH,HEIGHT= 960, 540

V_THRESH    = 5.0
IOU_THRESH  = 0.1
LOW_SPEED   = 2.0
LOW_IOU     = 0.05

# 2) INIT MODELS
yolo    = YOLO("models/yolov8n.pt")
tracker = DeepSort(max_age=60, n_init=2, nms_max_overlap=0.7)

# 3) PREPARE I/O
cap   = cv2.VideoCapture(VIDEO_PATH)
f     = open(LOG_CSV, "a", newline="", encoding="utf-8")
w     = csv.writer(f)
w.writerow(["start_time","end_time","attacker_id","victim_id","direction"])

prev_centers = {}
state        = "Idle"
attacker     = None
victim       = None
start_time   = None
direction    = None

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # DETECT
    results = yolo(frame, classes=[0], conf=0.25)
    dets = []
    for r in results:
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            if x2-x1<30 or y2-y1<30: continue
            dets.append(([x1,y1,x2-x1,y2-y1], float(b.conf[0]), "person", None))

    # TRACK
    tracks = tracker.update_tracks(dets, frame=frame) if dets else []

    # COMPUTE CENTERS & MOTIONS
    centers = {}
    motions = {}
    bboxes  = {}
    for t in tracks:
        if not t.is_confirmed(): continue
        tid = t.track_id
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        c = get_center((x1,y1,x2,y2))
        centers[tid] = c
        bboxes[tid]  = (x1,y1,x2,y2)
        if tid in prev_centers:
            motions[tid] = c - prev_centers[tid]
        else:
            motions[tid] = np.zeros(2, dtype=np.float32)

    ids = list(centers.keys())


    if len(ids) == 2:
        a, b = ids
        rel_speed = np.linalg.norm(motions[a] - motions[b])
        iou       = compute_iou(bboxes[a], bboxes[b])
        # Start fight
        if state == "Idle":
            if rel_speed > V_THRESH and iou > IOU_THRESH:
                v_ab = centers[b] - centers[a]
                projA = np.dot(motions[a],  v_ab)
                projB = np.dot(motions[b], -v_ab)
                if projA > projB:
                    attacker, victim = a, b
                    direction = (v_ab/np.linalg.norm(v_ab)).tolist()
                else:
                    attacker, victim = b, a
                    direction = (-v_ab/np.linalg.norm(v_ab)).tolist()
                start_time = now_str()
                state = "Fighting"
        # End fight
        elif state == "Fighting":
            if rel_speed < LOW_SPEED and iou < LOW_IOU:
                end_time = now_str()
                w.writerow([start_time, end_time, attacker, victim, direction])
                state      = "Idle"
                attacker   = victim = direction = start_time = None


    for t in tracks:
        if not t.is_confirmed(): continue
        tid = t.track_id
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        if state=="Fighting" and tid==attacker:
            role, color = "victim",   (0,255,0)
        elif state=="Fighting" and tid==victim:
            role, color = "attacker", (0,0,255)
        else:
            role, color = "stranger",(255,255,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"ID {tid} | {role}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    prev_centers = centers.copy()

    cv2.imshow("Fight Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
f.close()
cv2.destroyAllWindows()