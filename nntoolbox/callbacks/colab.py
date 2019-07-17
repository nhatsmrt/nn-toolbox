try:
    from google.colab import auth
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from oauth2client.client import GoogleCredentials
except:
    __all__ = []
else:
    from .callbacks import Callback
    from .checkpoint import ModelCheckpoint
    from ..utils import save_model
    import datetime as dt
    from typing import Dict, Any

    __all__ = ['ColabCheckpoint']


    class ColabCheckpoint(ModelCheckpoint):
        """
        Adapt from https://adventuresinmachinelearning.com/introduction-to-google-colaboratory/
        """
        def __init__(
                self, learner, drive_filename, filepath: str, monitor: str='loss',
                save_best_only: bool=True, mode: str='min', period: int=1
        ):
            super(ColabCheckpoint, self).__init__(learner, filepath, monitor,
                save_best_only, mode, period)
            self.drive_filename = drive_filename

            # Authenticate and create the PyDrive client.
            auth.authenticate_user()
            gauth = GoogleAuth()
            gauth.credentials = GoogleCredentials.get_application_default()
            self.drive = GoogleDrive(gauth)

        def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
            if self._save_best_only:
                epoch_metrics = logs['epoch_metrics']

                assert self._monitor in epoch_metrics
                self._metrics.append(epoch_metrics[self._monitor])

                if self._mode == "min":
                    if epoch_metrics[self._monitor] == min(self._metrics):
                        self.save()
                else:
                    if epoch_metrics[self._monitor] == max(self._metrics):
                        self.save()
            else:
                self.save()

            return False

        def save(self):
            save_model(self.learner._model, self._filepath)
            uploaded = self.drive.CreateFile({'title': self.drive_filename})
            uploaded.SetContentFile(self._filepath)
            uploaded.upload()
