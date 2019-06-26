import torchvision
from torch.nn import *
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from nntoolbox.optim import AdamW
# from adabound import AdaBound

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device
from nntoolbox.callbacks import *
from nntoolbox.metrics import Accuracy, Loss
from nntoolbox.vision.transforms import Cutout
from nntoolbox.vision.models import ImageClassifier

from functools import partial

from sklearn.metrics import accuracy_score


data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=ToTensor())
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_dataset.dataset.transform = Compose(
    [
        RandomHorizontalFlip(),
        RandomResizedCrop(size=32),
        Cutout(length=16, n_holes=1),
        ToTensor()
    ]
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
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

# layer_1 = ManifoldMixupModule(ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU))
# block_1 = ManifoldMixupModule(SEResNeXtShakeShake(in_channels=16, activation=nn.ReLU))

model = Sequential(
    ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3, activation=nn.ReLU),
    SEResNeXtShakeShake(in_channels=16, activation=nn.ReLU),
    # layer_1,
    # block_1,
    ConvolutionalLayer(
        in_channels=16, out_channels=32,
        activation=nn.Identity,
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
# print(model)


optimizer = AdamW(model.parameters(), weight_decay=0.0004)
# optimizer = Adam(model.parameters())
learner = SupervisedImageLearner(
    train_data=train_loader,
    val_data=val_loader,
    model=model,
    criterion=CrossEntropyLoss(),
    optimizer=optimizer,
    mixup=True
)

swa = StochasticWeightAveraging(model, average_after=50, update_every=100)
callbacks = [
    # ManifoldMixupCallback(learner=learner, modules=[layer_1, block_1]),
    Tensorboard(),
    ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=10),
    # LRSchedulerCB(CosineAnnealingLR(optimizer, 50)),
    # swa,
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
    EarlyStoppingCB(monitor='accuracy', mode='max', patience=20)
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
    ToPILImage(),
    RandomHorizontalFlip(),
    RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))
print("Test SWA:")
model = swa.get_averaged_model()
classifier = ImageClassifier(model, tta_transform=Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))

# total = 0
# accs = 0
# for images, labels in test_loader:
#     model.eval()
#     outputs = torch.argmax(model(images.to(get_device())), dim=1).cpu().detach().numpy()
#     labels = labels.cpu().numpy()
#     acc = accuracy_score(
#         y_true=labels,
#         y_pred=outputs
#     )
#
#     total += len(images)
#     accs += acc * len(images)
#
# print(accs / total)

