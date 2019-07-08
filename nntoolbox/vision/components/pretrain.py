from torch import nn
from .layers import InputNormalization
from torchvision.models import resnet18, vgg16_bn


class PretrainedModel(nn.Sequential):
    """
    based on https://github.com/chenyuntc/pytorch-book/blob/master/chapter8-%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB(Neural%20Style)/PackedVGG.py
    """
    def __init__(self, model=resnet18, embedding_size=128, fine_tune=False):
        super(PretrainedModel, self).__init__()
        model = model(pretrained=True)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False


        # model.fc = nn.Linear(model.fc.in_features, embedding_size)
        features = list(model.features)

        for ind in range(len(features)):
            self.add_module(
                "layer_" + str(ind),
                features[ind]
            )


class FeatureExtractor(nn.Module):
    """
    based on https://github.com/chenyuntc/pytorch-book/blob/master/chapter8-%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB(Neural%20Style)/PackedVGG.py
    """
    def __init__(self, model, mean=None, std=None, last_layer=None, fine_tune=True, device=None):
        super(FeatureExtractor, self).__init__()
        if mean is not None and std is not None:
            self._normalization = InputNormalization(mean=mean, std=std)
        else:
            self._normalization = nn.Identity()
        if not isinstance(model, nn.Module):
            model = model(pretrained=True)

        if device is not None:
            model.to(device)
            if self._normalization is not None:
                self._normalization.to(device)

        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        self._features = list(model.features)
        if last_layer is not None:
            self._features = self._features[:last_layer]
        self._features = nn.ModuleList(self._features)

    def forward(self, input, layers=None):
        input = self._normalization(input)
        op = []

        for ind in range(len(self._features)):
            input = self._features[ind](input)

            if layers is not None:
                if ind in layers:
                    op.append(input)

                if ind >= max(layers):
                    break
            else:
                if ind == len(self._features) - 1:
                    op.append(input)

        if len(op) == 1:
            return op[0]
        return op
