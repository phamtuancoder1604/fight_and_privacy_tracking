import numpy as np
from . import __init__  # placeholder for package context if needed

class FightFSM:
    def __init__(self, v_thresh=5.0, iou_thresh=0.1, low_speed=2.0, low_iou=0.05):
        self.v_thresh = v_thresh
        self.iou_thresh = iou_thresh
        self.low_speed = low_speed
        self.low_iou = low_iou
        self.state = "Idle"
        self.attacker = None
        self.victim = None
        self.start_time = None
        self.direction = None

    def maybe_start(self, ids, centers, motions, bboxes, now_str_fn, compute_iou):
        if self.state != "Idle" or len(ids) != 2:
            return False
        a, b = ids
        rel_speed = float(np.linalg.norm(motions[a] - motions[b]))
        iou = compute_iou(bboxes[a], bboxes[b])
        if rel_speed > self.v_thresh and iou > self.iou_thresh:
            v_ab = centers[b] - centers[a]
            projA = float(np.dot(motions[a],  v_ab))
            projB = float(np.dot(motions[b], -v_ab))
            if projA > projB:
                self.attacker, self.victim = a, b
                self.direction = (v_ab/np.linalg.norm(v_ab)).tolist() if np.linalg.norm(v_ab)>0 else [0,0]
            else:
                self.attacker, self.victim = b, a
                self.direction = (-v_ab/np.linalg.norm(v_ab)).tolist() if np.linalg.norm(v_ab)>0 else [0,0]
            self.start_time = now_str_fn()
            self.state = "Fighting"
            return True
        return False

    def maybe_end(self, ids, centers, motions, bboxes, now_str_fn, compute_iou):
        if self.state != "Fighting" or len(ids) != 2:
            return None
        a, b = ids
        rel_speed = float(np.linalg.norm(motions[a] - motions[b]))
        iou = compute_iou(bboxes[a], bboxes[b])
        if rel_speed < self.low_speed and iou < self.low_iou:
            end_time = now_str_fn()
            rec = [self.start_time, end_time, self.attacker, self.victim, self.direction]
            self.state = "Idle"
            self.attacker = self.victim = self.direction = self.start_time = None
            return rec
        return None
