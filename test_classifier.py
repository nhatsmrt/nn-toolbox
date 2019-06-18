import torchvision
import torch
from torch.nn import *
from torchvision.transforms import ToTensor
from torch.optim import *
from nntoolbox.optim import AdamW

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device
from nntoolbox.callbacks import Tensorboard, LossLogger, ModelCheckpoint, ReduceLROnPlateauCB
from nntoolbox.metrics import Accuracy, Loss

from sklearn.metrics import accuracy_score


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
    SEResidualBlockPreActivation(in_channels=64),
    ConvolutionalLayer(in_channels=64, out_channels=128, kernel_size=2, stride=2),
    SEResidualBlockPreActivation(in_channels=128),
    FeedforwardBlock(
        in_channels=128,
        out_features=10,
        pool_output_size=2,
        hidden_layer_sizes=(256,)
    )
)

optimizer = Adam(model.parameters())
learner = SupervisedImageLearner(
    train_data=train_loader,
    val_data=val_loader,
    model=model,
    criterion=CrossEntropyLoss(),
    optimizer=optimizer,
    use_scheduler=True,
    val_metric='accuracy',
)

callbacks = [
    Tensorboard(),
    ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=5),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
]
metrics = {
    "accuracy": Accuracy(),
    "loss": Loss()
}
learner.learn(
    n_epoch=500,
    callbacks=callbacks,
    metrics=metrics
)

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

