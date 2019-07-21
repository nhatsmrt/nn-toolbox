import torchvision
from torch.nn import *
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import *
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from nntoolbox.optim import AdamW
from torch.utils.data import random_split
# from adabound import AdaBound

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, LRFinder, get_first_batch, get_device
from nntoolbox.callbacks import *
from nntoolbox.metrics import Accuracy, Loss
from nntoolbox.vision.transforms import Cutout
from nntoolbox.vision.models import ImageClassifier, EnsembleImageClassifier
from nntoolbox.losses import SmoothedCrossEntropy
from nntoolbox.init import lsuv_init

import math


torch.backends.cudnn.benchmark=True

pretrained_model = resnet18(True)

# print(modules)
from nntoolbox.utils import cut_model, get_trainable_parameters

feature, head = cut_model(pretrained_model)
for param in feature.parameters():
    param.requires_grad = False

model = nn.Sequential(
    feature,
    FeedforwardBlock(
        in_channels=512,
        out_features=10,
        pool_output_size=2,
        hidden_layer_sizes=(256, 128)
    )
)
# print(model._modules['0']._modules[str(0)])

unfreezer = GradualUnfreezing([6, 4, 2, 0], 5)



# data = CIFAR10('data/', train=True, download=True, transform=ToTensor())
# train_size = int(0.8 * len(data))
# val_size = len(data) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
# train_dataset.dataset.transform = Compose(
#     [
#         RandomHorizontalFlip(),
#         RandomResizedCrop(size=32, scale=(0.95, 1.0)),
#         # Cutout(length=16, n_holes=1),
#         ToTensor()
#     ]
# )
#
# test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())


train_val_dataset = ImageFolder(
    'data/imagenette-160/train',
    transform=Compose([
        Resize((128, 128)),
        ToTensor()
    ])
)

test_dataset = ImageFolder(
    'data/imagenette-160/val',
    transform=Compose([
        Resize((128, 128)),
        ToTensor()
    ])
)

train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_dataset.dataset.transform = Compose(
    [
        RandomHorizontalFlip(),
        RandomResizedCrop(size=(128, 128), scale=(0.95, 1.0)),
        # Cutout(length=16, n_holes=1),
        ToTensor()
    ]
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# print(count_trainable_parameters(model)) # 14437816 3075928

optimizer = SGD(get_trainable_parameters(model), weight_decay=0.0001, lr=0.30, momentum=0.9)
learner = SupervisedImageLearner(
    train_data=train_loader,
    val_data=val_loader,
    model=model,
    criterion=SmoothedCrossEntropy().to(get_device()),
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
# lr_finder.find_lr(warmup=100, callbacks=[ToDeviceCallback()])

swa = StochasticWeightAveraging(learner, average_after=5025, update_every=670)
callbacks = [
    # ManifoldMixupCallback(learner=learner, modules=[layer_1, block_1]),
    ToDeviceCallback(),
    # MixedPrecisionV2(),
    # InputProgressiveResizing(initial_size=80, max_size=160, upscale_every=10, upscale_factor=math.sqrt(2)),
    # unfreezer,
    FreezeBN(),
    Tensorboard(),
    # ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=10),
    LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.10, T_max=335)),
    swa,
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
    ToPILImage(),
    RandomHorizontalFlip(),
    RandomResizedCrop(size=(128, 128), scale=(0.95, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))

print("Test SWA:")
model = swa.get_averaged_model()
classifier = ImageClassifier(model, tta_transform=Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    RandomResizedCrop(size=(128, 128), scale=(0.95, 1.0)),
    ToTensor()
]))
print(classifier.evaluate(test_loader))