from ax import optimize
from torch.nn import *
from torch.optim import *
import torchvision
import torch
from torchvision.transforms import ToTensor

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device

from sklearn.metrics import accuracy_score

class HyperparameterTuner:

    def __init__(self, train_data, val_data):
        return

def evaluation_function(lr):
    print("Evaluate at learning rate " + str(lr))

    data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=ToTensor())
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    model = Sequential(
        ConvolutionalLayer(in_channels=3, out_channels=16, kernel_size=3),
        SEResidualBlockPreActivation(in_channels=16),
        ConvolutionalLayer(in_channels=16, out_channels=32, kernel_size=2, stride=2),
        SEResidualBlockPreActivation(in_channels=32),
        ConvolutionalLayer(in_channels=32, out_channels=64, kernel_size=2, stride=2),
        AdaptiveAvgPool2d((4, 4)),
        Flatten(),
        Linear(in_features=1024, out_features=120),
        ReLU(),
        Dropout(),
        Linear(in_features=120, out_features=10)
    )

    criterion = CrossEntropyLoss()
    learner = SupervisedImageLearner(
        train_data=train_loader,
        val_data=val_loader,
        model=model,
        criterion=criterion,
        optimizer=Adam(model.parameters(), lr=lr),
        use_scheduler=True,
        val_metric='accuracy'
    )

    return learner.learn(n_epoch=1, print_every=1000, save_path="weights/model.pt")


best_parameters, best_values, experiment, model = optimize(
        parameters=[
          {
            "name": "lr",
            "type": "range",
            "bounds": [0.00001, 1.0],
          }
        ],
        evaluation_function=lambda p: evaluation_function(p["lr"]),
        minimize=False,
)

print(best_parameters)