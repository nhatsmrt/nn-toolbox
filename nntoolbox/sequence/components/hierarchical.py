"""
Reference:
https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
"""
from torch import nn, Tensor


__all__ = ['TimeDistributed']


class TimeDistributed(nn.Module):
    """
    Apply a module across time channel. Assume that 1st axis is the the major time axis, 2nd is the minor time axis,
    and 3rd axis is batch size. (UNTESTED)

    Example: major time axis is number of sentences per document; minor time axis is number of words per sentence;
    batch size is number of documents in a batch
    """
    def __init__(self, module: nn.Module):
        """
        :param module: takes as input a sequence data type with shape (seq_length, batch_size, *)
        """
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (doc_length, sent_length, batch_size, *)
        :return: (doc_length, sent_length, batch_size, *)
        """
        doc_length, sent_length, batch_size = input.shape[0], input.shape[1], input.shape[2]
        input = input.transpose(0, 1).contiguous() # (sent_length, doc_length, batch_size, *)
        new_shape = [sent_length, doc_length * batch_size] + list(input.shape[3:])
        input = input.view(new_shape)
        output = self.module(input)

        if isinstance(output, tuple):
            output = output[0]

        new_shape = [sent_length, doc_length, batch_size] + list(output.shape[2:])
        return output.view(new_shape).transpose(0, 1).contiguous()
