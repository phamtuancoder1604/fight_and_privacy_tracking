import numpy as np

def get_center(box):
    x1,y1,x2,y2 = box
    return np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)

def compute_iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def l2norm(v):
    return float(np.linalg.norm(v))
