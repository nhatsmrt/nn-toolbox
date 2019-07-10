from ..losses import FeatureLoss, StyleLoss, TotalVariationLoss, INStatisticsMatchingStyleLoss
from ..components import FeatureExtractor
from ..utils import tensor_to_pil, PairedDataset
from ...utils import save_model, load_model, is_valid
from ...callbacks import Callback, CallbackHandler

from torch.optim import Adam, Optimizer
import numpy as np
import torch
from torch.utils.data import DataLoader
from IPython.core.display import display # for display on notebook
from torch.nn import Module, MSELoss
from torch import Tensor

from typing import Iterable, Tuple


__all__ = ['StyleTransferLearner', 'MultipleStylesTransferLearner']


class StyleTransferLearner:
    def __init__(
            self, images:DataLoader, images_val:DataLoader, style_img:torch.Tensor,
            model:Module, feature_extractor:FeatureExtractor, content_layers, style_layers,
            style_weight:float, content_weight:float, total_variation_weight:float, device:torch.device
    ):
        self._model = model.to(device)
        self._images = images
        self._images_val = images_val
        self._style_img = style_img.to(device)
        self._style_weight = style_weight
        self._content_weight = content_weight
        self._total_variation_weight = total_variation_weight
        self._device = device

        self._feature_loss = FeatureLoss(feature_extractor, content_layers).to(device)
        self._style_loss = StyleLoss(feature_extractor, style_layers).to(device)
        self._total_variation_loss = TotalVariationLoss().to(device)
        self._optimizer = Adam(model.parameters())

    def learn(self, n_epoch, print_every=1, eval_every=1, draw=False, save_path=None, load_path=None):
        if load_path is not None:
            load_model(self._model, load_path)

        iter_cnt = 0
        for e in range(n_epoch):
            self._model.train()
            print("Epoch " + str(e))

            for batch_ndx, sample in enumerate(self._images):
                content_loss, style_loss, total_variation_loss = self.learn_one_iter(sample)

                if iter_cnt % print_every == 0:
                    print()
                    print()
                    print("Iter " + str(iter_cnt))
                    self.print_losses(content_loss, style_loss, total_variation_loss)

                iter_cnt += 1

            if e % eval_every == 0 and self._images_val is not None:
                self.evaluate(draw)
                if save_path is not None:
                    save_model(self._model, save_path)

    def learn_one_iter(self, images_batch: Tensor):
        '''
        :param images_batch: torch tensor, (N, C, H, W)
        :return: content loss, style loss, and total variation loss
        '''
        self._optimizer.zero_grad()
        content_loss, style_loss, total_variation_loss = self.compute_losses(images_batch)
        loss = content_loss + style_loss + total_variation_loss
        loss.backward()
        self._optimizer.step()

        return content_loss, style_loss, total_variation_loss

    @torch.no_grad()
    def evaluate(self, draw:bool):
        self._model.eval()

        for batch_ndx, sample in enumerate(self._images_val):
            content_loss, style_loss, total_variation_loss = self.compute_losses(sample)
            print()
            print()
            print()
            print("Evaluate: ")
            self.print_losses(content_loss, style_loss, total_variation_loss)

            if draw:
                random_ind = np.random.choice(len(sample))
                output = tensor_to_pil(self._model(sample[random_ind:random_ind+1].to(self._device)).cpu().detach())
                display(tensor_to_pil(sample[random_ind:random_ind+1].cpu()))
                display(output)

            break

    def compute_losses(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = self._model(images.to(self._device))
        content_loss = self._content_weight * self._feature_loss(outputs, images.to(self._device))
        style_loss = self._style_weight * self._style_loss(outputs, self._style_img)
        total_variation_loss = self._total_variation_weight * self._total_variation_loss(outputs)

        return content_loss, style_loss, total_variation_loss

    def print_losses(self, content_loss: Tensor, style_loss: Tensor, total_variation_loss: Tensor):
        print("Content loss: " + str(content_loss))
        print("Style loss: " + str(style_loss))
        print("Total variation loss: " + str(total_variation_loss))


class MultipleStylesTransferLearner:
    def __init__(
            self, content_style_imgs: DataLoader, content_style_val: DataLoader,
            model: Module, feature_extractor: FeatureExtractor, optimizer: Optimizer, style_layers,
            style_weight: float, content_weight: float, total_variation_weight: float, device: torch.device
    ):
        self._device = device
        self._feature_extractor = feature_extractor.to(device)
        self._train_data = self._content_style_imgs = content_style_imgs
        self._content_style_val = content_style_val
        self._model = model.to(self._device)
        self._content_weight = content_weight
        self._style_weight = style_weight
        self._total_variation_weight = total_variation_weight
        self._style_loss = INStatisticsMatchingStyleLoss(self._feature_extractor, style_layers).to(device)
        self._content_loss = MSELoss().to(device)
        self._total_variation_loss = TotalVariationLoss().to(device)
        self._optimizer = Adam(model.parameters()) if optimizer is None else optimizer

    def learn(self, n_iter: int, callbacks: Iterable[Callback], eval_every: int=1):
        print("Begin training")
        n_epoch = n_iter // eval_every
        self._cb_handler = CallbackHandler(self, callbacks=callbacks, n_epoch=n_epoch)
        self._cb_handler.on_train_begin()
        for iter in range(n_iter):
            self._model.train()
            for content_batch, style_batch in self._content_style_imgs:
                data = self._cb_handler.on_batch_begin({"content": content_batch, "style": style_batch}, True)
                content_batch, style_batch = data["content"], data["style"]
                self.learn_one_iter(content_batch, style_batch)
            if iter % eval_every == 0:
                stop_training = self.evaluate()
                if stop_training:
                    break
        self._cb_handler.on_train_end()

    def learn_one_iter(self, content_batch: Tensor, style_batch: Tensor):
        content_loss, style_loss, total_variation_loss = self.compute_losses(content_batch, style_batch)
        total_loss = content_loss + style_loss + total_variation_loss

        if self._cb_handler.on_backward_begin():
            total_loss.backward()
        if self._cb_handler.after_backward():
            self._optimizer.step()
            if self._cb_handler.after_step():
                self._optimizer.zero_grad()
            logs = {
                'content_loss': content_loss,
                'style_loss': style_loss, 'total_variation_loss': total_variation_loss,
                'loss': total_loss
            }
            self._cb_handler.on_batch_end(logs)

    @torch.no_grad()
    def evaluate(self):
        self._model.eval()
        for content_batch, style_batch in self._content_style_val:
            data = self._cb_handler.on_batch_begin({"content": content_batch, "style": style_batch}, False)
            content_batch, style_batch = data["content"], data["style"]

            self._model.set_style(style_batch)
            styled_imgs = self._model(content_batch).cpu().detach()
            outputs = self._cb_handler.after_outputs(
                {"content_batch": content_batch, "style_batch": style_batch, "styled_imgs": styled_imgs}, False
            )
            content_batch, style_batch, styled_imgs = \
                outputs["content_batch"], outputs["style_batch"], outputs["styled_imgs"]
            imgs = torch.cat((content_batch.cpu(), style_batch.cpu(), styled_imgs), dim=0)
            tag = \
                ["content_" + str(i) for i in range(len(content_batch))] + \
                ["style_" + str(i) for i in range(len(content_batch))] +  \
                ["styled_" + str(i) for i in range(len(content_batch))]
            return self._cb_handler.on_epoch_end({"draw": imgs, "tag": tag})

    def compute_losses(self, content_batch: Tensor, style_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        self._model.set_style(style_batch)
        t = self._model.style_encode(content_batch) # t = AdaIn(f(c), f(s))
        output = self._model.decode(t) # g(t)
        fgt = self._model.encode(output) # f(g(t))

        outputs = self._cb_handler.after_outputs(
            {"output": output, "fgt": fgt, "t": t, "style_batch": style_batch},
            True
        )
        output, fgt, t, style_batch = outputs["output"], outputs["fgt"], outputs["t"], outputs["style_batch"]

        content_loss = self._content_weight * self._content_loss(t, fgt)
        style_loss = self._style_weight * self._style_loss(output, style_batch)
        total_variation_loss = self._total_variation_weight * self._total_variation_loss(output)
        losses = self._cb_handler.after_losses(
            {"content": content_loss, "style": style_loss, "tv": total_variation_loss}, True
        )
        content_loss, style_loss, total_variation_loss = losses["content"], losses["style"], losses["tv"]
        return content_loss, style_loss, total_variation_loss


