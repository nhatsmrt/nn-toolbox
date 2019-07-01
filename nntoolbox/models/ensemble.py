from typing import List, Optional
import numpy as np


__all__ = ['Ensemble']


class Ensemble:
    def __init__(self, models, model_weights: Optional[List[float]]=None):
        assert len(models) > 0
        if model_weights is not None:
            assert len(models) == len(model_weights)
            assert len(model_weights.shape) == 1

        self.models = models
        if model_weights is not None:
            self.model_weights = np.array(model_weights) / sum(model_weights)
        else:
            self.model_weights = np.array([1 / len(self.models) for _ in range(len(self.models))])
