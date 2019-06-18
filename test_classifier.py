import torchvision
import torch
from torch.nn import *
from torchvision.transforms import ToTensor
from torch.optim import *
from nntoolbox.optim import AdamW

from nntoolbox.vision.components import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.utils import load_model, get_device
from nntoolbox.callbacks import Tensorboard, LossLogger, ModelCheckpoint, ReduceLROnPlateauCB, EarlyStoppingCB
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
    FeedforwardBlock(
        in_channels=64,
        out_features=10,
        pool_output_size=4,
        hidden_layer_sizes=(512,)
    )
)

optimizer = Adam(model.parameters(), weight_decay=0.0004)
learner = SupervisedImageLearner(
    train_data=train_loader,
    val_data=val_loader,
    model=model,
    criterion=CrossEntropyLoss(),
    optimizer=optimizer,
    mixup=True
)

callbacks = [
    Tensorboard(),
    ReduceLROnPlateauCB(optimizer, monitor='accuracy', mode='max', patience=10),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
    EarlyStoppingCB(monitor='accuracy', mode='max', patience=15)
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

