import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class _ResidualBlockNoBN(nn.Sequential):
    '''
    Residual Block without the final Batch Normalization layer
    '''

    def __init__(self, in_channels):
        super(_ResidualBlockNoBN, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                ConvolutionalLayer(in_channels, in_channels, 3, padding=1),
                nn.Conv2d(
                    in_channels=in_channels, out_channels=in_channels,
                    kernel_size=3, stride=1, padding=1,
                ),
                nn.LeakyReLU()
            )
        )

    def forward(self, input):
        return super(_ResidualBlockNoBN, self).forward(input) + input

class ResidualBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                _ResidualBlockNoBN(in_channels),
                nn.BatchNorm2d(in_channels)
            )
        )


class ConvolutionalLayer(nn.Sequential):
    '''
    Simple convolutional layer: input -> conv2d -> activation -> batch norm 2d
    '''
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, padding=0,
            bias=False, activation=nn.LeakyReLU
    ):
        super(ConvolutionalLayer, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                ),
                activation(inplace=True),
                nn.BatchNorm2d(num_features=out_channels)
            )
        )

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class ResizeConvolutionalLayer(nn.Module):
    '''
    Upsample the image (using an interpolation algorithm), then pass to a conv layer
    '''
    def __init__(self, in_channels, out_channels, mode='bilinear'):
        super(ResizeConvolutionalLayer, self).__init__()
        self._mode = mode
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, input, out_h, out_w):
        upsampled = F.interpolate(input, size=(out_h, out_w), mode=self._mode)
        return self._modules["conv"](upsampled)

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, activation):
        super(DenseLayer, self).__init__()

        self.add_module(
            "main",
            nn.Sequential(
                nn.BatchNorm2d(num_features = in_channels),
                activation(inplace = True),
                ConvolutionalLayer(
                    in_channels = in_channels,
                    out_channels = growth_rate,
                    kernel_size = 1,
                    stride = 1,
                    bias=False,
                    activation=activation
                ),
                nn.Conv2d(
                    in_channels = growth_rate,
                    out_channels = growth_rate,
                    kernel_size = 3,
                    stride = 1,
                    padding=1,
                    bias=False
                )
            )
        )

    def forward(self, input):
        return torch.cat((input, super(DenseLayer, self).forward(input)), dim = 1)

class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, num_layers, activation=nn.ReLU):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module(
                "DenseLayer_" + str(i),
                DenseLayer(
                    in_channels = in_channels + growth_rate * i,
                    growth_rate = growth_rate,
                    activation=activation
                )
            )

class Reshape(nn.Module):
    def forward(self, input, new_shape):
        return input.view(new_shape)

# Input: N * C * H * W
def spatial_pyramid_pool(input, op_sizes, pool_layer=nn.MaxPool2d):
    ops = []
    batch_size = input.shape[0]
    inp_h = input.shape[2]
    inp_w = input.shape[3]


    for size in op_sizes:
        pool = pool_layer(
            kernel_size=(torch.ceil(torch.Tensor([inp_h / size])), torch.ceil(torch.Tensor([inp_w / size]))),
            stride=(torch.floor(torch.Tensor([inp_h / size])), torch.floor(torch.Tensor([inp_w / size])))
        )
        ops.append(pool(input).view(batch_size, -1))

    # for op in ops:
    #     print(op.shape)
    return torch.cat(ops, dim = -1)

class SpatialPyramidPool(nn.Module):
    def __init__(self, op_sizes, pool_layer = nn.MaxPool2d):
        super(SpatialPyramidPool, self).__init__()
        self._op_sizes = op_sizes
        self._pool_layer = pool_layer

    def forward(self, input):
        return spatial_pyramid_pool(input, self._op_sizes, self._pool_layer)



class GlobalAveragePool(nn.Module):
    def __init__(self):
        super(GlobalAveragePool, self).__init__()
    def forward(self, input):
        return torch.mean(torch.mean(input, dim = -1), dim = -1)



# based on https://github.com/chenyuntc/pytorch-book/blob/master/chapter8-%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB(Neural%20Style)/PackedVGG.py
class PretrainedModel(nn.Sequential):
    def __init__(self, model=resnet18, embedding_size=128, fine_tune=False):
        super(PretrainedModel, self).__init__()
        model = model(pretrained = True)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False


        model.fc = nn.Linear(model.fc.in_features, embedding_size)

        self.add_module(
            "model",
            model
        )

class L2NormalizationLayer(nn.Module):
    def __init__(self):
        super(L2NormalizationLayer, self).__init__()

    def forward(self, input):
        return F.normalize(input, dim=-1, p=2)
