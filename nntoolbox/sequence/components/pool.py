from torch import nn, Tensor


__all__ = ['MaxOverTime']


class MaxOverTime(nn.Module):
    """
    Implement max pooling over time

    References:

        Yoon Kim. "Convolutional Neural Networks for Sentence Classification."
        https://aclweb.org/anthology/D14-1181

        Ronan Collobert et al. "Natural Language Processing (Almost) from Scratch."
        http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf
    """
    def __init__(self, batch_first: bool=False):
        super(MaxOverTime, self).__init__()
        self.batch_first = batch_first

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (batch_size, seq_len, *) if batch_first, else (seq_len, batch_size, *)
        :return: (batch_size, *)
        """
        return input.max(1)[0] if self.batch_first else input.max(0)[0]
