import queue
import threading


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator):
        super().__init__()
        self.queue = queue.Queue(4)
        self.generator = generator

        self.daemon = True
        self.start()


    def __iter__(self):
        return self


    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)