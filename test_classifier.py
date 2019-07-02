import torchvision
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from nntoolbox.optim import AdamW
# from adabound import AdaBound

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device, LRFinder
from nntoolbox.callbacks import *
from nntoolbox.metrics import Accuracy, Loss
from nntoolbox.vision.transforms import Cutout
from nntoolbox.vision.models import ImageClassifier, EnsembleImageClassifier
from nntoolbox.losses import SmoothedCrossEntropy

from functools import partial

from sklearn.metrics import accuracy_score

torch.backends.cudnn.benchmark=True


data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=ToTensor())
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_dataset.dataset.transform = Compose(
    [
        RandomHorizontalFlip(),
        RandomResizedCrop(size=32, scale=(0.95, 1.0)),
        # Cutout(length=16, n_holes=1),
        ToTensor()
    ]
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)
# kernel = partial(PolynomialKernel, dp=3, cp=2.0)


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


class StandAloneMultiheadAttentionLayer(nn.Sequential):
    def __init__(
            self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=3,
            activation=nn.ReLU, normalization=nn.BatchNorm2d
    ):
        layers = [
            StandAloneMultiheadAttention(
                num_heads=num_heads,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            activation(),
            normalization(num_features=out_channels),
        ]
        super(StandAloneMultiheadAttentionLayer, self).__init__(*layers)


class SEResNeXtShakeShakeAttention(ResNeXtBlock):
    def __init__(self, num_heads, in_channels, reduction_ratio=16, cardinality=2, activation=nn.ReLU,
                 normalization=nn.BatchNorm2d):
        super(SEResNeXtShakeShakeAttention, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        StandAloneMultiheadAttentionLayer(
                            num_heads=num_heads,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=7,
                            activation=activation,
                            normalization=normalization
                        ),
                        StandAloneMultiheadAttentionLayer(
                            num_heads=num_heads,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=7,
                            activation=activation,
                            normalization=normalization
                        ),
                        SEBlock(in_channels, reduction_ratio)
                    ) for _ in range(cardinality)
                ]
            ),
            use_shake_shake=True
        )


# layer_1 = ManifoldMixupModule(ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU))
# block_1 = ManifoldMixupModule(SEResNeXtShakeShake(in_channels=16, activation=nn.ReLU))

model = Sequential(
    ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU),
    SEResNeXtShakeShake(in_channels=16, activation=nn.ReLU),
    # layer_1,
    # block_1,
    ConvolutionalLayer(
        in_channels=16, out_channels=32,
        activation=nn.ReLU,
        kernel_size=2, stride=2
    ),
    SEResNeXtShakeShake(in_channels=32),
    StandAloneMultiheadAttentionLayer(
        num_heads=8, in_channels=32, out_channels=64,
        kernel_size=7, stride=2
    ),
    SEResNeXtShakeShakeAttention(num_heads=8, in_channels=64),
    StandAloneMultiheadAttentionLayer(
        num_heads=8, in_channels=64, out_channels=128,
        kernel_size=7, stride=2
    ),
    SEResNeXtShakeShakeAttention(num_heads=8, in_channels=128),
    FeedforwardBlock(
        in_channels=128,
        out_features=10,
        pool_output_size=2,
        hidden_layer_sizes=(512, 256)
    )
)

optimizer = SGD(model.parameters(), weight_decay=0.0001, lr=0.06, momentum=0.9)
# optimizer = Adam(model.parameters())
learner = SupervisedImageLearner(
    train_data=train_loader,
    val_data=val_loader,
    model=model,
    criterion=SmoothedCrossEntropy(),
    optimizer=optimizer,
    mixup=True
)

# lr_finder = LRFinder(
#     model=model,
#     train_data=train_loader,
#     criterion=SmoothedCrossEntropy(),
#     optimizer=partial(SGD, lr=0.074, weight_decay=0.0001, momentum=0.9),
#     device=get_device()
# )
# lr_finder.find_lr(warmup=100)

swa = StochasticWeightAveraging(learner, average_after=11200, update_every=3200)
callbacks = [
    # ManifoldMixupCallback(learner=learner, modules=[layer_1, block_1]),
    Tensorboard(),
    # ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=10),
    LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.02, T_max=1600)),
    swa,
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
]
metrics = {
    "accuracy": Accuracy(),
    "loss": Loss()
}
final = learner.learn(
    n_epoch=100,
    callbacks=callbacks,
    metrics=metrics,
    final_metric='accuracy'
)
print(final)
load_model(model=model, path="weights/model.pt")
classifier = ImageClassifier(model, tta_transform=Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    RandomResizedCrop(size=32, scale=(0.95, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))

print("Test SWA:")
model = swa.get_averaged_model()
classifier = ImageClassifier(model, tta_transform=Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    RandomResizedCrop(size=32, scale=(0.95, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))
