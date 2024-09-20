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
    # a dictionary of BoW features for each token in the hypothesis and the premise
    unigram_features = dict()
    premise = ex["sentence1"]
    hypothesis = ex["sentence2"]
    for token in premise + hypothesis:
        # increment the count of the token in the dictionary
        unigram_features[token] = unigram_features.get(token, 0) + 1
    return unigram_features


def extract_custom_features(ex):
    """Returns features for a given pair of hypothesis and premise by performing the following:
    1. Extracting the unigrams and bigrams in the hypothesis and the premise.
    2. Scaling the features by the following two scales:
        - sentence frequency: log(number of sentences [= 2 in this case] / number of sentences containing the token)
        - token frequency: log(inverse of the fraction of the total number of tokens in the combined sentences that are the token)
    3. Multiplying the features by the two scales.

    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of features for x.

    Example:
        "I love it", "I hate it" --> {
            "I": 2 * log(2/2.001) * log(10/2.001) = -0.001
            "it": 2 * log(2/2.001) * log(10/2.001) = -0.001
            "love": 1 * log(2/1.001) * log(10/1.001) = 1.5930
            "hate": 1 * log(2/1.001) * log(10/1.001) = 1.593
            ("I", "love"): 1 * log(2/1.001) * log(10/1.001) = 1.593
            ("love", "it"): 1 * log(2/1.001) * log(10/1.001) = 1.593
            ("I", "hate"): 1 * log(2/1.001) * log(10/1.001) = 1.593
            ("hate", "it"): 1 * log(2/1.001) * log(10/1.001) = 1.593
        }
    """
    custom_features = dict()

    premise = ex["sentence1"]
    hypothesis = ex["sentence2"]
    premise_bigrams = premise + list(nltk.ngrams(premise, 2))
    hypothesis_bigrams = hypothesis + list(nltk.ngrams(hypothesis, 2))
    combined = premise_bigrams + hypothesis_bigrams

    for token in combined:
        # increment the count of the token in the dictionary
        custom_features[token] = custom_features.get(token, 0) + 1

    total_tokens = len(combined)
    for token in custom_features:
        # we can scale the feature by multiplying the following two scales
        # intuition: we want the weight of a token to be maximum when it appears in the fewest number of sentences and
        # it appears as few times as possible whenever it appears in a sentence. This is because the more unique a token,
        # the more it can help in distinguishing the sentences.
        sentence_frequency = np.log(
            2
            / (int(token in premise_bigrams) + int(token in hypothesis_bigrams) + 1e-3)
        )
        token_frequency = np.log(total_tokens / (custom_features[token] + 1e-3))
        custom_features[token] *= sentence_frequency * token_frequency
    return custom_features


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
    # extract features from the training and validation data
    train_features = list(map(feature_extractor, train_data))
    valid_features = list(map(feature_extractor, valid_data))

    # create a vocabulary of all the features
    vocabulary = set()
    for feature in train_features:
        vocabulary.update(feature.keys())

    # initialize the weights of the features. We opt for zero initialization here.
    weights = {word: 0.0 for word in vocabulary}
    total_train_examples = len(train_data)
    total_valid_examples = len(valid_data)

    for epoch in range(num_epochs):
        # initialize the loss for the training and validation data
        training_loss = 0.0
        validation_loss = 0.0
        for i, ex in enumerate(train_data):
            label = ex["gold_label"]
            # predict the label using the weights
            prediction = predict(weights, train_features[i])
            # compute the loss and the gradient
            loss = -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))
            gradient = (prediction - label) * np.array(list(train_features[i].values()))
            for idx, word in enumerate(train_features[i]):
                # update the weights using the gradient (SGD)
                weights[word] -= learning_rate * gradient[idx]
            training_loss += loss
        # compute the average training loss
        training_loss /= total_train_examples

        for i, ex in enumerate(valid_data):
            label = ex["gold_label"]
            # predict the label using the weights
            prediction = predict(weights, valid_features[i])
            # compute the loss
            validation_loss += -(
                label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
            )
        # compute the average validation loss
        validation_loss /= total_valid_examples

        # print the training and validation loss for each epoch
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
    # compute a dictionary mapping words to indices
    word2ind = {word: idx for idx, word in enumerate(vocabulary)}
    # initialize the co-occurrence matrix
    co_mat = np.zeros((vocabulary_size, vocabulary_size))
    for i, token in enumerate(tokens):
        # the window size comprises tokens in the range [i - window_size, i + window_size], excluding the token itself
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
            if i != j:
                # increment the count of the co-occurring words
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
    # compute the SVD of the co-occurrence matrix. We opt for the full SVD here to obtain both U and S matrices.
    # we set hermitian=True because the matrix is real and symmetric (since a co-occurrence matrix is always symmetric)
    u, s, vt = np.linalg.svd(co_mat, full_matrices=True, hermitian=True)
    # we take only the first embed_size columns of U and multiply it with a diagonal matrix of the first embed_size singular values
    embeddings = u[:, :embed_size] @ np.diag(s[:embed_size])
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
    # compute a dictionary mapping indices to words
    ind2word = {ind: word for word, ind in word2ind.items()}
    # normalize the embeddings for quicker computation
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_word_embedding = embeddings[word_ind]
    # compute the similarity scores of the query word with all the words
    # since we have normalized the embeddings, the cosine similarity is equivalent to the dot product
    similarity_scores = np.array(
        list(map(lambda v: np.dot(query_word_embedding, v), embeddings))
    )
    # sort the similarity scores in descending order and return the top k words
    # we exclude the query word itself by starting the slicing from the second element
    top_k_indices = similarity_scores.argsort()[::-1][1 : k + 1]
    return list(map(lambda x: ind2word[x], top_k_indices))
