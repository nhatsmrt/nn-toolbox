from ..losses import FeatureLoss, StyleLoss, TotalVariationLoss
from ...utils import compute_num_batch
from torch.optim import Adam
import numpy as np
from tqdm import trange

class StyleTransferLearner:
    def __init__(
            self, images, style_img, content_img,
            model, feature_extractor, feature_layers, style_layers,
            style_weight, content_weight, total_variation_weight, device
    ):
        self._model = model.to(device)
        self._images = images.to(device)
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

    def learn(self, n_epoch, batch_size):
        n_batch = compute_num_batch(len(self._images), batch_size)
        indices = np.arange(len(self._images))

        for e in range(n_epoch):
            self._model.train()
            print("Epoch " + str(e))
            np.random.shuffle(indices)

            for i in trange(n_batch):
                idx = indices[i * batch_size:(i + 1) * batch_size]
                images_batch = self._images[idx]
                content_loss, style_loss, total_variation_loss = self.learn_one_iter(images_batch)
                print()
                print()
                print("Content loss: " + str(content_loss))
                print("Style loss loss: " + str(style_loss))
                print("Total variation loss: " + str(total_variation_loss))




    def learn_one_iter(self, images_batch):
        self._optimizer.zero_grad()

        outputs = self._model(images_batch)
        content_loss = self._content_weight * self._feature_loss(outputs, self._content_img)
        style_loss = self._style_weight * self._style_loss(outputs, self._style_img)
        total_variation_loss = self._total_variation_weight * self._total_variation_loss(outputs)

        loss = content_loss + style_loss + total_variation_loss
        loss.backward()
        self._optimizer.step()

        return content_loss, style_loss, total_variation_loss
