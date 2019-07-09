"""
Add a progress bar to learner
Adapt from fastai course 2 v3 notebook
"""
from .callbacks import Callback
from fastprogress import master_bar, progress_bar
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader


__all__ = ['ProgressBarCB']


class ProgressBarCB(Callback):
    n_epoch: int
    
    def __init__(self, dataloader: Optional[DataLoader]=None):
        self.dataloader = dataloader
        
    def on_train_begin(self):
        if self.dataloader is None:
            self.dataloader = self.learner._train_data

        self.master_bar = master_bar(range(self.n_epoch))
        self.master_bar.on_iter_begin()
        self.set_progress(self.dataloader, 0)

    def on_batch_end(self, logs: Dict[str, Any]):
        self.progress_bar.update(logs["iter_cnt"] % len(self.dataloader))
        self.master_bar.child.comment = f'Iterations'

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        self.set_progress(self.dataloader, logs["epoch"] + 1)
        return False

    def on_train_end(self): self.master_bar.on_iter_end()

    def set_progress(self, data: DataLoader, epoch: int):
        self.progress_bar = progress_bar(data, parent=self.master_bar, auto_update=False)
        self.master_bar.update(epoch)
        self.master_bar.first_bar.comment = f'Epochs'
