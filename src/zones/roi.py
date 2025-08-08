# Define ROI polygons/zones if you want to filter detections by area.
class RoiFilter:
    def __init__(self, polys=None):
        self.polys = polys or []
