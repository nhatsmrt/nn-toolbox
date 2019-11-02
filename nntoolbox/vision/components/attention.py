import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


__all__ = ['SAGANAttention', 'StandAloneSelfAttention', 'StandAloneMultiheadAttention']


class SAGANAttention(nn.Module):
    """
    Implement SAGAN attention module.

    References:

        Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena. "Self-Attention Generative Adversarial Networks."
        https://arxiv.org/pdf/1805.08318.pdf
    """
    def __init__(self, in_channels: int, reduction_ratio: int=8):
        assert in_channels % reduction_ratio == 0

        super().__init__()
        self.transform = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(in_channels // reduction_ratio) * 3,
            kernel_size=1, bias=False
        )
        self.softmax = nn.Softmax(dim=1)
        self.op_transform = nn.Conv2d(
            in_channels=in_channels // reduction_ratio,
            out_channels=in_channels,
            kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, _, h, w = input.shape
        transformed = self.transform(input)
        key, query, value = transformed.chunk(3, 1)

        attention_scores = key.view((batch_size, -1, h * w)).permute(0, 2, 1).bmm(
            query.view((batch_size, -1, h * w))
        )
        attention_weights = self.softmax(attention_scores)
        output = value.view(batch_size, value.shape[1], -1).bmm(attention_weights)
        output = output.view(batch_size, output.shape[1], h, w)
        output = self.op_transform(output)
        return self.scale * output + input


class StandAloneSelfAttention(nn.Conv2d):
    """
    A single head of Stand-Alone Self-Attention for Vision Model

    References:

        Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, Jonathon Shlens.
        "Stand-Alone Self-Attention in Vision Models." https://arxiv.org/pdf/1906.05909.pdf.
    """
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size, stride=1,
            padding: int=0, dilation: int=1,
            bias: bool=True, padding_mode: str='zeros'
    ):
        assert out_channels % 2 == 0
        super(StandAloneSelfAttention, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, 1,
            bias, padding_mode
        )
        self.weight = None
        self.bias = None
        self.transform = nn.Conv2d(in_channels, out_channels * 3, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.rel_h = nn.Embedding(num_embeddings=self.kernel_size[0], embedding_dim=out_channels // 2)
        self.rel_w = nn.Embedding(num_embeddings=self.kernel_size[1], embedding_dim=out_channels // 2)
        self.h_range = nn.Parameter(torch.arange(0, self.kernel_size[0])[:, None], requires_grad=False)
        self.w_range = nn.Parameter(torch.arange(0, self.kernel_size[1])[None, :], requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, _, inp_h, inp_w = input.shape
        output_h, output_w = self.compute_output_shape(inp_h, inp_w)
        if self.padding_mode == 'circular':
            expanded_padding = [(self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2]
            input = F.pad(input, expanded_padding, mode='circular')
            padding = 0
        else:
            padding = self.padding

        transformed = self.transform(input)
        key, query, value = transformed.chunk(3, 1)

        key_uf = F.unfold(
            key, kernel_size=self.kernel_size, dilation=self.dilation,
            padding=padding, stride=self.stride
        ).view(
            batch_size, self.out_channels, self.kernel_size[0], self.kernel_size[1], -1
        )[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2, :]
        query_uf = F.unfold(
            query, kernel_size=self.kernel_size, dilation=self.dilation,
            padding=padding, stride=self.stride
        ).view(batch_size, self.out_channels, self.kernel_size[0] * self.kernel_size[1], -1)
        value_uf = F.unfold(
            value, kernel_size=self.kernel_size, dilation=self.dilation,
            padding=padding, stride=self.stride
        ).view(batch_size, self.out_channels, self.kernel_size[0] * self.kernel_size[1], -1)

        rel_embedding = self.get_rel_embedding()[None, :, :, None]
        logits = (key_uf[:, :, None, :] * (query_uf + rel_embedding)).sum(1, keepdim=True)

        attention_weights = self.softmax(logits)

        output = (attention_weights * value_uf).sum(2).view(batch_size, -1, output_h, output_w)
        return output

    def compute_output_shape(self, height, width):
        def compute_shape_helper(inp_dim, padding, kernel_size, dilation, stride):
            return np.floor(
                (inp_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            ).astype(np.uint32)
        return (
            compute_shape_helper(height, self.padding[0], self.kernel_size[0], self.dilation[0], self.stride[0]),
            compute_shape_helper(width, self.padding[1], self.kernel_size[1], self.dilation[1], self.stride[1]),
        )

    def get_rel_embedding(self) -> Tensor:
        h_embedding = self.rel_h(self.h_range).repeat(1, self.kernel_size[1], 1)
        w_embedding = self.rel_w(self.w_range).repeat(self.kernel_size[0], 1, 1)
        return torch.cat((h_embedding, w_embedding), dim=-1).view(-1, self.out_channels).transpose(0, 1)

    def to(self, *args, **kwargs):
        self.h_range.to(*args, **kwargs)
        self.w_range.to(*args, **kwargs)
        super().to(*args, **kwargs)


class StandAloneMultiheadAttention(nn.Module):
    """
    Stand-Alone Multihead Self-Attention for Vision Model

    References:

        Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, Jonathon Shlens.
        "Stand-Alone Self-Attention in Vision Models." https://arxiv.org/pdf/1906.05909.pdf.
    """
    def __init__(
            self, num_heads: int, in_channels: int, out_channels: int, kernel_size, stride=1,
            padding: int=0, dilation: int=1,
            bias: bool=True, padding_mode: str='zeros'
    ):
        assert out_channels % num_heads == 0
        super(StandAloneMultiheadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [
                StandAloneSelfAttention(
                    in_channels, out_channels // num_heads, kernel_size, stride,
                    padding, dilation, bias, padding_mode
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, input: Tensor) -> Tensor:
        heads = [head(input) for head in self.heads]
        return torch.cat(heads, dim=1)
