"""
Add a progress bar to learner
Adapt from fastai course 2 v3 notebook
"""
from .callbacks import Callback
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from typing import Dict, Any


class ProgressBar(Callback):
    def on_train_begin(self):
        self.master_bar = master_bar()
        self.master_bar.on_iter_begin()

    def on_train_end(self): self.master_bar.on_iter_end()

    def on_batch_end(self, logs: Dict[str, Any]): return

    def set_progress(self, epoch):
        self.progress_bar = progress_bar()
        self.master_bar.update(epoch)
