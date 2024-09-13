import json
import collections
import argparse
import random

import nltk
import numpy as np

from util import *

random.seed(42)


def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    unigram_features = dict()  # a dictionary that maps tokens to their frequency
    premise = ex["sentence1"]
    hypothesis = ex["sentence2"]
    for token in premise + hypothesis:
        unigram_features[token] = unigram_features.get(token, 0) + 1
    return unigram_features


def extract_custom_features(ex):
    """Returns features for a given pair of hypothesis and premise by performing the following:
    1. Removes stop words from both sentences.

    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of features for x.

    Example:
        "I love it", "I hate it" --> {"love":1, "hate":1}
    """
    custom_features = dict()
    try:
        stopwords = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        stopwords = nltk.corpus.stopwords.words("english")

    premise = ex["sentence1"]
    hypothesis = ex["sentence2"]

    # premise = list(filter(lambda x: x not in stopwords, list(map(str.lower, ex["sentence1"]))))
    # hypothesis = list(filter(lambda x: x not in stopwords, list(map(str.lower, ex["sentence1"]))))

    # premise = list(filter(lambda x: x not in stopwords, ex["sentence1"]))
    # hypothesis = list(filter(lambda x: x not in stopwords, ex["sentence2"]))

    # premise = list(map(str.lower, ex["sentence1"]))
    # hypothesis = list(map(str.lower, ex["sentence2"]))

    # n = 3
    # premise = list(nltk.ngrams(premise, n))
    # hypothesis = list(nltk.ngrams(hypothesis, n))

    combined = premise + hypothesis
    total_tokens = len(combined)

    for token in combined:
        custom_features[token] = custom_features.get(token, 0) + 1
    for token in custom_features:
        scale_1 = np.log(2 / (int(token in premise) + int(token in hypothesis)))
        scale_2 = np.log((total_tokens + 1) / (combined.count(token) + 1))
        custom_features[token] *= scale_1 * scale_2
    return custom_features


def compute_vocabulary(features):
    """Compute the vocabulary of the data.
    Parameters:
        features : [{str: int}]
            A list of feature dictionaries.
    Returns:
        A list of words (str)
    """
    vocabulary = set()
    for feature in features:
        vocabulary.update(feature.keys())
    return list(vocabulary)


def learn_predictor(
    train_data, valid_data, feature_extractor, learning_rate, num_epochs
):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    train_features = list(map(feature_extractor, train_data))
    valid_features = list(map(feature_extractor, valid_data))
    vocabulary = compute_vocabulary(train_features)
    weights = {word: 0.0 for word in vocabulary}
    total_examples = len(train_data)

    for epoch in range(num_epochs):
        training_loss = 0.0
        validation_loss = 0.0
        for i, ex in enumerate(train_data):
            label = ex["gold_label"]
            prediction = predict(weights, train_features[i])
            loss = -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))
            gradient = (prediction - label) * np.array(list(train_features[i].values()))
            for idx, word in enumerate(train_features[i]):
                weights[word] -= learning_rate * gradient[idx]
            training_loss += loss
        training_loss /= total_examples

        for i, ex in enumerate(valid_data):
            label = ex["gold_label"]
            prediction = predict(weights, valid_features[i])
            validation_loss += -(
                label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
            )
        validation_loss /= len(valid_data)

        print(
            f"Epoch {epoch}: training loss = {training_loss}, validation loss = {validation_loss}"
        )

    return weights


def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    vocabulary = set(tokens)
    vocabulary_size = len(vocabulary)
    word2ind = {word: idx for idx, word in enumerate(vocabulary)}
    co_mat = np.zeros((vocabulary_size, vocabulary_size))
    for i, token in enumerate(tokens):
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
            if i != j:
                co_mat[word2ind[token]][word2ind[tokens[j]]] += 1

    return word2ind, co_mat


def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    u, s, vh = np.linalg.svd(co_mat, full_matrices=True)
    embeddings = np.dot(u[:, :embed_size], np.diag(s[:embed_size]))
    return embeddings


def top_k_similar(word_ind, embeddings, word2ind, k=10, metric="dot"):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    ind2word = {idx: word for word, idx in word2ind.items()}
    word_embedding = embeddings[word_ind]
    metrics = {
        "dot": lambda embedding1, embedding2: np.dot(embedding1, embedding2),
        "cosine": lambda embedding1, embedding2: np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ),
    }
    similarity_metric = metrics[metric]
    similarity_scores = np.array(
        [similarity_metric(word_embedding, embeddings[i]) for i in range(len(embeddings))]
    )
    top_k_word_indices = np.argsort(similarity_scores)[::-1][:k]
    top_k_words = [ind2word[i] for i in top_k_word_indices]
    return top_k_words

