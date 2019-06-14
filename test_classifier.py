import torchvision
import torch
from torch.nn import *
from torchvision.transforms import ToTensor
from torch.optim import *

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device

from sklearn.metrics import accuracy_score


data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=ToTensor())
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

# Conv2d(1, 16, 5, stride=1, padding=2),
# ReLU(),
# MaxPool2d(kernel_size=2, stride=2),
# Conv2d(16, 32, 5, stride=1, padding=2),
# ReLU(),
# MaxPool2d(kernel_size=2, stride=2),

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
    optimizer=Adam(model.parameters()),
    use_scheduler=True,
    val_metric='accuracy',
    use_tb=True
)

learner.learn(n_epoch=500, print_every=1000, save_path="weights/model.pt")
load_model(model=model, path="weights/model.pt")

total = 0
accs = 0
for images, labels in test_loader:
    model.eval()
    outputs = torch.argmax(model(images.to(get_device())), dim=1).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(
        y_true=labels,
        y_pred=outputs
    )

    total += len(images)
    accs += acc * len(images)

print(accs / total)



# for images, labels in val_loader:
#     print(labels.shape)
#     print(model(images).shape)
#     break


