from ..losses import FeatureLoss, StyleLoss, TotalVariationLoss
from ..components import FeatureExtractor
from ..utils import tensor_to_pil
from torch.optim import Adam
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import Module


class StyleTransferLearner:
    def __init__(
            self, images:DataLoader, images_val:DataLoader, style_img:torch.Tensor, content_img:torch.Tensor,
            model:Module, feature_extractor:FeatureExtractor, feature_layers, style_layers,
            style_weight:float, content_weight:float, total_variation_weight:float, device:torch.device
    ):
        self._model = model.to(device)
        self._images = images
        self._images_val = images_val
        self._style_img = style_img.to(device)
        self._content_img = content_img.to(device)
        self._style_weight = style_weight
        self._content_weight = content_weight
        self._total_variation_weight = total_variation_weight
        self._device = device

        self._feature_loss = FeatureLoss(feature_extractor, feature_layers).to(device)
        self._style_loss = StyleLoss(feature_extractor, style_layers).to(device)
        self._total_variation_loss = TotalVariationLoss().to(device)
        self._optimizer = Adam(model.parameters())

    def learn(self, n_epoch, print_every=1, eval_every=1, draw=False):
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

    def learn_one_iter(self, images_batch:torch.Tensor):
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
                output = tensor_to_pil(self._model(sample[random_ind:random_ind+1].to(self._device)))
                output.show()

    def compute_losses(self, images):
        outputs = self._model(images.to(self._device))
        content_loss = self._content_weight * self._feature_loss(outputs, self._content_img)
        style_loss = self._style_weight * self._style_loss(outputs, self._style_img)
        total_variation_loss = self._total_variation_weight * self._total_variation_loss(outputs)

        return content_loss, style_loss, total_variation_loss

    def print_losses(self, content_loss, style_loss, total_variation_loss):
        print("Content loss: " + str(content_loss))
        print("Style loss loss: " + str(style_loss))
        print("Total variation loss: " + str(total_variation_loss))