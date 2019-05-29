from torch import nn
from torchvision.models import resnet18

# based on https://github.com/chenyuntc/pytorch-book/blob/master/chapter8-%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB(Neural%20Style)/PackedVGG.py
class PretrainedModel(nn.Sequential):
    def __init__(self, model=resnet18, embedding_size=128, fine_tune=False):
        super(PretrainedModel, self).__init__()
        model = model(pretrained = True)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False


        model.fc = nn.Linear(model.fc.in_features, embedding_size)

        self.add_module(
            "model",
            model
        )
