import torchvision
from torch.nn import *
from torchvision.datasets import CIFAR10
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device
from nntoolbox.callbacks import *
from nntoolbox.metrics import Accuracy, Loss
from nntoolbox.vision.models import ImageClassifier
from nntoolbox.losses import SmoothedCrossEntropy

torch.backends.cudnn.benchmark=True


data = CIFAR10('data/', train=True, download=True, transform=ToTensor())
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_dataset.dataset.transform = Compose(
    [
        RandomHorizontalFlip(),
        RandomResizedCrop(size=32, scale=(0.95, 1.0)),
        ToTensor()
    ]
)

test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Number of batches per epoch " + str(len(train_loader)))


class SEResNeXtShakeShake(ResNeXtBlock):
    def __init__(self, in_channels, reduction_ratio=16, cardinality=2, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(SEResNeXtShakeShake, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        ConvolutionalLayer(
                            in_channels, in_channels // 4, kernel_size=1, padding=0,
                            activation=activation, normalization=normalization
                        ),
                        ConvolutionalLayer(
                            in_channels // 4, in_channels, kernel_size=3, padding=1,
                            activation=activation, normalization=normalization
                        ),
                        SEBlock(in_channels, reduction_ratio)
                    ) for _ in range(cardinality)
                ]
            ),
            use_shake_shake=True
        )


model = Sequential(
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
    FeedforwardBlock(
        in_channels=64,
        out_features=10,
        pool_output_size=2,
        hidden_layer_sizes=(256, 128)
    )
).to(get_device())

optimizer = SGD(model.parameters(), weight_decay=0.0001, lr=0.094, momentum=0.9)
learner = SupervisedImageLearner(
    train_data=train_loader,
    val_data=val_loader,
    model=model,
    criterion=SmoothedCrossEntropy().to(get_device()),
    optimizer=optimizer,
    mixup=True
)


callbacks = [
    ToDeviceCallback(),
    Tensorboard(),
    LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.024, T_max=405)),
    GradualLRWarmup(min_lr=0.024, max_lr=0.094, duration=810),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
]

metrics = {
    "accuracy": Accuracy(),
    "loss": Loss()
}

final = learner.learn(
    n_epoch=500,
    callbacks=callbacks,
    metrics=metrics,
    final_metric='accuracy'
)


print(final)
load_model(model=model, path="weights/model.pt")
classifier = ImageClassifier(model, tta_transform=Compose([
    RandomHorizontalFlip(),
    RandomResizedCrop(size=32, scale=(0.95, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))

print("Test SWA:")
model = swa.get_averaged_model()
classifier = ImageClassifier(model, tta_transform=Compose([
    RandomHorizontalFlip(),
    RandomResizedCrop(size=32, scale=(0.95, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))
