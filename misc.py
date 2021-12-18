import time

class Queue:
    def __init__(self, size):
        self.queue = []
        self._size = size

    @property
    def qsize(self):
        return len(self.queue)

    def enqueue(self, *e):
        if len(e) < 1:
            pass
        else:
            self.queue.extend(e)
            while self.qsize > self._size:
                self.dequeue()

    def dequeue(self):
        self.queue.pop(0)

    def find(self, value, place=lambda x: x):
        for i, e in enumerate(self.queue):
            if place(e) == value:
                return i
        return None

    def __len__(self):
        return self.qsize

    def __getitem__(self, idx):
        return self.queue[idx]

    def __repr__(self):
        return str(self.queue)

class History(Queue):
    def __init__(self, size):
        Queue.__init__(self, size)

    def recent(self, size):
        try:
            return self.queue[-size:]
        except:
            raise ValueError

class FPS:
    def __init__(self):
        self.prev = 0

    def __str__(self):
        now = time.perf_counter()
        fps = 1/(now - self.prev)
        self.prev = now
        return str(fps)