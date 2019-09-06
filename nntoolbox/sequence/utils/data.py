from torchtext.vocab import Vocab
import spacy
from typing import List
import multiprocessing as mp
from functools import partial


__all__ = ['id_to_text', 'text_to_id', 'tokenize_sentence']


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
