from src.utils.geometry import compute_iou, get_center

def test_iou_zero():
    assert abs(compute_iou((0,0,10,10),(20,20,30,30))) < 1e-6

def test_center():
    cx, cy = get_center((0,0,10,10))
    assert cx == 5 and cy == 5
