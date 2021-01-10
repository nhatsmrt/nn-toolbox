from warnings import warn
from torchtext.vocab import Vocab
from typing import List


__all__ = ['id_to_text', 'text_to_id']


def id_to_text(sequence, vocab: Vocab) -> List[str]:
    """
    Convert a sequence of id to corresponding text

    :param sequence:
    :param vocab: vocab object
    :return: text (array form)
    """
    return [vocab.itos[token] for token in sequence]


def text_to_id(sequence: List[str], vocab: Vocab) -> List[int]:
    """
    Convert a sequence of string to corresponding list of numeric tokens

    :param sequence: list of string
    :param vocab: vocab object
    :return: list of tokens
    """
    return [vocab.stoi[word] for word in sequence]


try:
    import spacy
    import multiprocessing as mp
    from functools import partial


    __all__ += ['tokenize_sentence', 'tokenize_sentences']



    def tokenize_sentence(sentence: str, model) -> List[str]:
        """Tokenize a single sentence"""
        return model.tokenizer(sentence)


    def tokenize_fn(ind: int, sentences: List[str], model):
      tokenized = tokenize_sentence(sentences[ind], model)
      return tokenized


    def tokenize_sentences(sentences: List[str], model, num_workers: int=1) -> List[List[str]]:
        """Tokenize a list of sentences"""
        if num_workers > 1:
            pool = mp.Pool(processes=num_workers)
            return list(pool.map(partial(tokenize_fn, sentences, model), range(len(sentences))))

        return [tokenize_sentence(sentence, model) for sentence in sentences]
except ImportError:
    warn("spacy is not installed so certain methods in nntoolbox.sequence.utils.data will not be usable")
