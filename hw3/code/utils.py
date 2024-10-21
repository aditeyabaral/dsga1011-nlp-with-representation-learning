import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # A mappping of qwerty keyboard keys to their neighbors
    qwerty_neighbors = {
        "a": ["q", "w", "s", "z"],
        "b": ["v", "g", "h", "n"],
        "c": ["x", "d", "f", "v"],
        "d": ["s", "e", "r", "f", "x", "c"],
        "e": ["w", "s", "d", "r"],
        "f": ["d", "r", "t", "g", "v", "c"],
        "g": ["f", "t", "y", "h", "v", "b"],
        "h": ["g", "y", "u", "j", "b", "n"],
        "i": ["u", "j", "k", "o"],
        "j": ["h", "u", "i", "k", "n", "m"],
        "k": ["j", "i", "o", "l", "m"],
        "l": ["k", "o", "p"],
        "m": ["n", "j", "k"],
        "n": ["b", "h", "j", "m"],
        "o": ["i", "k", "l", "p"],
        "p": ["o", "l"],
        "q": ["a", "w"],
        "r": ["e", "d", "f", "t"],
        "s": ["a", "w", "e", "d", "x", "z"],
        "t": ["r", "f", "g", "y"],
        "u": ["y", "h", "j", "i"],
        "v": ["c", "f", "g", "b"],
        "w": ["q", "a", "s", "e"],
        "x": ["z", "s", "d", "c"],
        "y": ["t", "g", "h", "u"],
        "z": ["a", "s", "x"],
    }

    original_text = example["text"]
    words = word_tokenize(original_text)

    for i, word in enumerate(words):
        # Lowercase the word
        if random.random() > 0.5:
            words[i] = word.lower()

        # Replace some words with synonyms only if it is a noun
        if (
            random.random() > 0.5
            and len(word) > 3
            and wordnet.synsets(word)
            and wordnet.synsets(word)[0].pos() == "n"
        ):
            syns = wordnet.synsets(word)
            if len(syns) > 0:
                syn = syns[0].lemmas()[0].name()
                words[i] = syn

        # Replace some letters with typos
        if random.random() > 0.5:
            new_word = ""
            for letter in word:
                if random.random() > 0.85:
                    to_upper = False
                    if letter.isupper():
                        letter = letter.lower()
                        to_upper = True
                    sampled_letter = random.choice(
                        qwerty_neighbors.get(letter, [letter])
                    )
                    if to_upper:
                        sampled_letter = sampled_letter.upper()
                    new_word += sampled_letter
                else:
                    new_word += letter
            words[i] = new_word

    example["text"] = TreebankWordDetokenizer().detokenize(words)

    ##### YOUR CODE ENDS HERE ######

    return example
