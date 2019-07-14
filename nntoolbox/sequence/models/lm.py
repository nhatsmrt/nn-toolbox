from torch import nn, Tensor


__all__ = ['LanguageModel']


class LanguageModel(nn.Module):
    """
    General Language Model (UNTESTED):

    p(x_t|x_{<t}) = prod_{t' < t} p(x_{t'} | x_{<t'})
    """
    def __init__(self, encoder: nn.Module, head: nn.Module):
        """
        :param encoder: encoding the hidden states of words up to now
        :param head: compute score for each word given the input
        """
        super(LanguageModel, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (seq_length, batch_size, input_dim)
        :return: (seq_length, batch_size, vocab_size)
        """
        output = self.encoder(input) # (seq_length, batch_size, output_dim)
        return self.classifier_head(output) # (seq_length, batch_size, vocab_size)

    def get_encoder(self) -> nn.Module:
        """
        Extract the encoder for downstream task

        :return: encoder
        """
        return self.encoder

    def replace_head(self, head: nn.Module):
        """
        Replace the head (e.g for fine-tuning on another domain)

        :param head:
        """
        self.head = head
