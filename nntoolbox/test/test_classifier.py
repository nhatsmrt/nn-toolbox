import torchvision
from torch.nn import *
from torchvision.datasets import ImageFolder, CIFAR10
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

from functools import partial

from sklearn.metrics import accuracy_score
from .prog_bar import progress_bar_test


def run_classifier_test():
    print("Starting classifier test")
    # progress_bar_test()
    torch.backends.cudnn.benchmark = True

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
    # kernel = partial(PolynomialKernel, dp=3, cp=2.0)


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

    class SEResNeXtShakeShake(ResNeXtBlock):
        def __init__(self, in_channels, reduction_ratio=16, cardinality=2, activation=nn.ReLU,
                     normalization=nn.BatchNorm2d):
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
                            # ConvolutionalLayer(
                            #     in_channels // 4, in_channels, kernel_size=1, padding=0,
                            #     activation=activation, normalization=normalization
                            # ),
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
                            ConvolutionalLayer(
                                in_channels=in_channels,
                                out_channels=in_channels // 2,
                                kernel_size=1,
                                activation=activation,
                                normalization=normalization
                            ),
                            StandAloneMultiheadAttentionLayer(
                                num_heads=num_heads,
                                in_channels=in_channels // 2,
                                out_channels=in_channels // 2,
                                kernel_size=3,
                                activation=activation,
                                normalization=normalization
                            ),
                            ConvolutionalLayer(
                                in_channels=in_channels // 2,
                                out_channels=in_channels,
                                kernel_size=1,
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
        ConvolutionalLayer(
            in_channels=128, out_channels=256,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=256),
        ConvolutionalLayer(
            in_channels=256, out_channels=512,
            kernel_size=2, stride=2
        ),
        SEResNeXtShakeShake(in_channels=512),
        # SEResNeXtShakeShakeAttention(num_heads=8, in_channels=512),
        FeedforwardBlock(
            in_channels=512,
            out_features=10,
            pool_output_size=2,
            hidden_layer_sizes=(256, 128)
        )
    ).to(get_device())

    # lsuv_init(module=model, input=get_first_batch(train_loader, callbacks = [ToDeviceCallback()])[0])

    # print(count_trainable_parameters(model)) # 14437816 3075928

    optimizer = SGD(model.parameters(), weight_decay=0.0001, lr=0.30, momentum=0.9)
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
        InputProgressiveResizing(initial_size=80, max_size=160, upscale_every=10, upscale_factor=math.sqrt(2)),
        # MixedPrecisionV2(),
        Tensorboard(),
        NaNWarner(),
        # ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=10),
        LRSchedulerCB(CosineAnnealingLR(optimizer, eta_min=0.10, T_max=335)),
        swa,
        LossLogger(),
        ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
        ProgressBarCB()
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
