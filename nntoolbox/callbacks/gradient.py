from .callbacks import Callback
from torch.nn.utils import clip_grad_value_, clip_grad_norm_


__all__ = ['GradientValueClipping', 'GradientNormClipping']


# UNTESTED
class GradientValueClipping(Callback):
    def __init__(self, clip_value: float):
        """
        :param clip_value: range of allowed gradient: (-clip, clip)
        """
        self.clip_value = clip_value

    def after_backward(self) -> bool:
        clip_grad_value_(self.learner._model.parameters(), self.clip_value)
        return True


# UNTESTED
class GradientNormClipping(Callback):
    def __init__(self, max_norm: float, norm_type=2):
        """
        :param clip_value: range of allowed gradient: (-clip, clip)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def after_backward(self) -> bool:
        clip_grad_norm_(self.learner._model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type)
        return True
