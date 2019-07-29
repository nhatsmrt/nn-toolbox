from typing import Dict, Any


__all__ = ['Metric', 'Loss']


class Metric:
    _best: float

    def __call__(self, logs: Dict[str, Any]): pass

    def get_best(self) -> float: return self._best


class Loss(Metric):
    def __init__(self):
        self._best = float('inf')

    def __call__(self, logs: Dict[str, Any]) -> float:
        if logs['loss'] <= self._best:
            self._best = logs['loss']

        return logs["loss"]
