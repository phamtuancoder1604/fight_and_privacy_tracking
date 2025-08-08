from src.events.fight_event import FightFSM
from src.utils.geometry import compute_iou
import numpy as np

def now_str():
    return "2025-01-01 00:00:00"

def test_fsm_idle():
    fsm = FightFSM()
    rec = fsm.maybe_end([], {}, {}, {}, now_str, compute_iou)
    assert rec is None
