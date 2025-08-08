import time

class FpsMeter:
    def __init__(self):
        self.t0 = time.time()
        self.count = 0
        self.fps = 0.0

    def update(self):
        self.count += 1
        now = time.time()
        dt = now - self.t0
        if dt >= 1.0:
            self.fps = self.count / dt
            self.t0 = now
            self.count = 0
        return self.fps
