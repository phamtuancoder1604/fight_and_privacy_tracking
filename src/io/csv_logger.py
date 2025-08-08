import csv, os

class CsvLogger:
    def __init__(self, path, header=None):
        self.path = path
        self.header = header or []
        self._ensure_header()

    def _ensure_header(self):
        exists = os.path.exists(self.path)
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if not exists and self.header:
            self.w.writerow(self.header)

    def write_row(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
