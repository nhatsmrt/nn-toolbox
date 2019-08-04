import torch
from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F
import numpy as np


__all__ = ['GlobalSelfAttention', 'StandAloneSelfAttention', 'StandAloneMultiheadAttention']


# UNTESTED
class GlobalSelfAttention(nn.Module):
    """
    Implement attention module as described by:

    https://arxiv.org/pdf/1805.08318.pdf
    """
    def __init__(self, in_channels: int, reduction_ratio: int=8):
        super(GlobalSelfAttention, self).__init__()
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
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, input: Tensor) -> Tensor:
        batch_size, in_channels, h, w = input.shape
        transformed = self.transform(input)
        n_channel_each = transformed.shape[1] // 3
        key, query, value = (
            transformed[:, :n_channel_each, :, :],
            transformed[:, n_channel_each:2 * n_channel_each, :, :],
            transformed[:, 2 * n_channel_each:, :, :]
        )
        # key, query, value = self.key_transform(input), self.query_transform(input), self.value_transform(input)
        attention_scores = key.view((batch_size, -1, h * w)).permute(0, 2, 1).bmm(
            query.view((batch_size, -1, h * w))
        )
        attention_weights = self.softmax(attention_scores)
        output = value.view(batch_size, value.shape[1], -1).bmm(attention_weights)
        output = output.view(batch_size, output.shape[1], h, w)
        output = self.op_transform(output)
        return self.scale * output + input


# UNTESTED
class StandAloneSelfAttention(nn.Conv2d):
    """
    Implement a single head of self attention:

    https://arxiv.org/pdf/1906.05909v1.pdf
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
        self.key_transform = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.query_transform = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.value_transform = nn.Conv2d(in_channels, out_channels, 1, bias=False)
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

        # start = time.time()
        key, query, value = self.key_transform(input), self.query_transform(input), self.value_transform(input)
        # end = time.time()
        # print("Transform time: " + str(end - start))

        # start = time.time()
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
        # end = time.time()
        # print("Unfold time: " + str(end - start))

        # start = time.time()
        rel_embedding = self.get_rel_embedding()[None, :, :, None]
        logits = (key_uf[:, :, None, :] * (query_uf + rel_embedding)).sum(1, keepdim=True)
        # end = time.time()
        # print("Find logit time: " + str(end - start))

        # start = time.time()
        attention_weights = self.softmax(logits)
        # end = time.time()
        # print("Softmax time: " + str(end - start))

        # start = time.time()
        output = (attention_weights * value_uf).sum(2).view(batch_size, -1, output_h, output_w)
        # end = time.time()
        # print("Output time: " + str(end - start))
        # print()

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


# UNTESTED
class StandAloneMultiheadAttention(nn.Module):
    """
    Implement stand alone multihead attention
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


#INCOMPLETE
class AttentionAugmentedConv2D(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int,
            key_depth: int, num_heads: int, attentional_channels: int,
            relative: bool, h: int=4, w: int=4
    ):
        assert key_depth % num_heads == 0
        assert attentional_channels % num_heads == 0
        super(AttentionAugmentedConv2D, self).__init__()

        self._fin = in_channels
        self._nh = num_heads
        self._dk = key_depth
        self._dv = attentional_channels
        self._fout = out_channels
        self._relative = relative
        if relative:
            self._h = h
            self._w = w
            self._adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(h, w))
            self._key_rel_w = nn.Parameter(torch.randn((2 * w - 1, key_depth), requires_grad=True))
            self._key_rel_h = nn.Parameter(torch.randn((2 * h - 1, key_depth), requires_grad=True))


        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels - attentional_channels,
            kernel_size=kernel_size,
            padding=1
        )
        self._conv_attn = nn.Conv2d(
            in_channels=attentional_channels,
            out_channels=attentional_channels,
            kernel_size=1
        )
        self._conv_qkv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * key_depth + attentional_channels,
            kernel_size=1
        )
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        if self._relative:
            input = self._adaptive_pool(input)

        batch_size, fin, h, w = input.shape
        conv_out = self._conv(input)
        flat_q, flat_k, flat_v = self.compute_flat_qkv(input)
        logits = torch.bmm(
            flat_q.view(batch_size * self._nh, -1, flat_q.shape[3]).permute(0, 2, 1),
            flat_k.view(batch_size * self._nh, -1, flat_k.shape[3]),
        ).view(batch_size, self._nh, h * w, h * w)

        if self._relative:
            h_rel_logits, w_rel_logits = self.relative_logits(flat_q)
            logits = logits + h_rel_logits + w_rel_logits

        weights = self._softmax(logits) # batch_size, nh, hw, hw
        attn_out = torch.bmm(
            flat_v.view(batch_size * self._nh, -1, h * w),
            weights.view(-1, h * w, h * w).permute(0, 2, 1),
        ).view(batch_size, -1, h, w)
        attn_out = self._conv_attn(attn_out)

        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param input: (batch_size, fin, h, w)
        :return: (batch_size, nh, dkh, hw), (batch_size, nh, dkh, hw), (batch_size, nh, dvh, hw)
        """
        batch_size, fin, h, w = input.shape
        qkv = self._conv_qkv(input)
        q = qkv[:, :self._dk, :, :]
        k = qkv[:, self._dk:2 * self._dk, :, :]
        v = qkv[:, 2 * self._dk:, :, :]

        q = q.view(batch_size, self._nh, -1, h * w).contiguous()
        k = k.view(batch_size, self._nh, -1, h * w).contiguous()
        v = v.view(batch_size, self._nh, -1, h * w).contiguous()

        q = q * ((self._dk // self._nh) ** (-0.5))
        return q, k, v

    def relative_logits(self, query):
        return (
            self.relative_logits_1d(query, self._key_rel_h, "h"),
            self.relative_logits_1d(query, self._key_rel_w, "w")
        )

    def relative_logits_1d(self, query, rel_k, case):

        return


# INCOMPLETE
class Attention2D(nn.Module):
    def __init__(self):
        super(Attention2D, self).__init__()

    def forward(self, keys: Tensor, queries: Tensor, values: Tensor) -> Tensor:
        """
        :param keys: (batch_size, key_dim, H, W)
        :param queries: (batch_size, query_dim, H, W)
        :param values: (batch_size, value_dim, H, W)
        :return: (batch_size, value_dim, H, W)
        """

    def compute_attention_weights(self, keys, queries):
        """
        :param keys: (batch_size, key_dim, H, W)
        :param queries: (batch_size, query_dim, H, W)
        :return: (batch_size,
        """
