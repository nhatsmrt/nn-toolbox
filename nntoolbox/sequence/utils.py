from __future__ import unicode_literals, print_function, division
import unicodedata
import numpy as np


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def create_mask(inputs, pad_token):
    '''
    Create a binary mask to indicate whether a token is pad or not
    :param inputs: (seq_len, batch_size)
    :return: mask: (seq_len, batch_size)
    '''
    return (inputs != pad_token)

def get_lengths(mask):
    '''
    Return a 1D array indicating the length of each sequence in batch
    :param mask: binary mask indicating whether an element is pad token (seq_len, batch_size)
    :return: lengths (n_batch)
    '''
    return np.sum(mask, axis=0).astype(np.uint8)
