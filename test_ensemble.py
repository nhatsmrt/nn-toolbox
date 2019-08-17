import torchvision
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
# from adabound import AdaBound

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device, LRFinder
from nntoolbox.callbacks import *
from nntoolbox.metrics import Accuracy, Loss
from nntoolbox.vision.transforms import Cutout
from nntoolbox.vision.models import ImageClassifier, EnsembleImageClassifier
from nntoolbox.losses import SmoothedCrossEntropy
from nntoolbox.ensembler import CVEnsembler


data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=ToTensor())
test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)


class SEResNeXtShakeShake(ResNeXtBlock):
    def __init__(self, in_channels, reduction_ratio=16, cardinality=2, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(SEResNeXtShakeShake, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        SEBlock(in_channels, reduction_ratio)
                    ) for _ in range(cardinality)
                ]
            ),
            use_shake_shake=True
        )


def model_fn():
    return Sequential(
        ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU),
        SEResNeXtShakeShake(in_channels=16, activation=nn.ReLU),
        ConvolutionalLayer(
            in_channels=16, out_channels=32,
            activation=nn.ReLU,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=32),
        ConvolutionalLayer(
            in_channels=32, out_channels=64,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=64),
        ConvolutionalLayer(
            in_channels=64, out_channels=128,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=128),
        FeedforwardBlock(
            in_channels=128,
            out_features=10,
            pool_output_size=2,
            hidden_layer_sizes=(512, 256)
        )
    )


def learn_fn(train_data, val_data, model, save_path):
    optimizer = SGD(model.parameters(), weight_decay=0.0001, lr=0.094, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True)
    learner = SupervisedImageLearner(
        train_data=train_loader,
        val_data=val_loader,
        model=model,
        criterion=SmoothedCrossEntropy(),
        optimizer=optimizer,
        mixup=True
    )

    callbacks = [
        ToDeviceCallback(),
        Tensorboard(),
        LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.024, T_max=1600)),
        LossLogger(),
        ModelCheckpoint(learner=learner, filepath=save_path, monitor='accuracy', mode='max'),
        # EarlyStoppingCB(monitor='accuracy', mode='max', patience=20)
    ]
    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss()
    }
    final = learner.learn(
        n_epoch=2,
        callbacks=callbacks,
        metrics=metrics,
        final_metric='accuracy'
    )
    print(final)
    load_model(model=model, path="weights/model.pt")
    return model

ensembler = CVEnsembler(data, path="weights/", n_model=3, learn_fn=learn_fn, model_fn=model_fn)
ensembler.learn()

ensemble_model = EnsembleImageClassifier([ImageClassifier(model) for model in ensembler.get_models()])
print(ensemble_model.evaluate(test_loader))

