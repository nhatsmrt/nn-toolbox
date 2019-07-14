from torchtext.vocab import Vocab


__all__ = ['id_to_text']


def id_to_text(sequence, vocab: Vocab):
    """
    Convert a sequence of id to corresponding text

    :param sequence:
    :param vocab: vocab object
    :return: text (array form)
    """
    ret = []
    for token in sequence:
        ret.append(vocab.itos[token])
    return ret
