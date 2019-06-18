from .callbacks import Callback


class MixUpCallback(Callback):
    def on_batch_begin(self, images, labels, train):
        if train:
            return
