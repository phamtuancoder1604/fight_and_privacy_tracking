import cv2

class VideoWriter:
    def __init__(self, out_path, width, height, fps=30):
        self.out_path = out_path
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        else:
            self.vw = None

    def write(self, frame):
        if self.vw is not None:
            self.vw.write(frame)

    def release(self):
        if self.vw is not None:
            self.vw.release()
