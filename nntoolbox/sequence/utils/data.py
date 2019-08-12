from torchtext.vocab import Vocab
import spacy
from typing import List
import multiprocessing as mp
from functools import partial


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


def tokenize_sentence(sentence: str, model=spacy.load('en')) -> List[str]:
    """Tokenize a single sentence"""
    return model.tokenizer(sentence)


def tokenize_fn(ind: int, sentences: List[str], model=spacy.load('en')):
  tokenized = tokenize_sentence(sentences[ind], model)
  return tokenized


def tokenize_sentences(sentences: List[str], model=spacy.load('en'), num_workers: int=1) -> List[List[str]]:
    """Tokenize a list of sentences"""
    if num_workers > 1:
        pool = mp.Pool(processes=num_workers)
        return list(pool.map(partial(tokenize_fn, sentences, model), range(len(sentences))))

    return [tokenize_sentence(sentence, model) for sentence in sentences]
