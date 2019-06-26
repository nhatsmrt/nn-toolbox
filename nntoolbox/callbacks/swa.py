from .callbacks import Callback
from ..utils import copy_model
from typing import Dict, Any


class StochasticWeightAveraging(Callback):
    def __init__(self, model, update_every=1, average_after=20):
        '''
        :param model: the model currently being trained
        :param update_every: how many epochs between each average update
        :param update_after: the first epoch to start averaging
        '''
        self._model = model
        self.model_swa = copy_model(model)
        self._update_every = update_every
        self._average_after = average_after

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if logs["epoch"] >= self._average_after and logs["epoch"] % self._update_every == 0:
            n_model = (logs["epoch"] - self._average_after) / self._update_every
            w1 = self._model.named_parameters()
            w2 = self.model_swa.named_parameters()

            dict_params2 = dict(w2)
            for name1, param1 in w1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_((param1.data + n_model * dict_params2[name1].data) / (n_model + 1))

            self.model_swa.load_state_dict(dict_params2)
        return False

    def get_averaged_model(self):
        '''
        Return the post-training average model
        :return: the averaged model
        '''
        return self.model_swa