import numpy as np


class AnnealingSchedule:
    def __init__(self, start=1.0, end=0.1, decay_steps=500, _delay=0):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.annealed_value = np.linspace(start, end, decay_steps)
        self._delay = _delay

    def get_value(self, ts):
        return self.annealed_value[min(max(1, ts - self._delay), self.decay_steps) - 1]  # deal with the edge case


def _test_scheduler():
    scheduler = AnnealingSchedule(start=1.0, end=0.01, decay_steps=10)
    for i in range(1, 10):
        print(i, scheduler.get_value(ts=i))

    scheduler = AnnealingSchedule(start=1.0, end=0.01, decay_steps=10, _delay=5)
    for i in range(1, 10 + 5):
        print(i, scheduler.get_value(ts=i))


if __name__ == '__main__':
    _test_scheduler()
