import logging
import math
from typing import List

import numpy as np
from sklearn.metrics import f1_score

from activation import softmax
from loss import cross_entropy_loss


def xavier_initialization(shape: tuple) -> np.ndarray:
    scale = 1 / max(1., (sum(shape)) / len(shape))
    limit = math.sqrt(3.0 * scale)
    return np.random.uniform(-limit, limit, size=shape)


class SkipGram(object):
    def __init__(self, n_input_dim: int, alpha: float, epochs: int, n_emb_dim: int, window_size: int):
        self.n_input_dim = n_input_dim
        self.alpha = alpha
        self.epochs = epochs
        self.n_emb_dim = n_emb_dim
        self.window_size = window_size

        # initialize weights
        self._hidden_weight = xavier_initialization(shape=(self.n_input_dim, self.n_emb_dim))  # (34, emb_dim)
        self.out_weight = xavier_initialization(shape=(self.n_input_dim, self.n_emb_dim))

        self.losses = []
        self.final_loss = []
        self.result = []

    def _label_encode_word_set(self, word_set) -> List[List[int]]:
        # convert the type of words to int from string
        word2idx = {str(word): idx for idx, word in enumerate(range(self.n_input_dim))}
        return [
            [word2idx[word] for word in words]
            for words in word_set
        ]

    def forward(self, word_set, window_size) -> None:
        losses = []
        score_diffs = []
        for target_idx, target_word in enumerate(word_set):
            # convert a target word to one-hot vector
            target_one_hot = np.zeros(34, np.float_)
            target_one_hot[target_word] = 1.0

            # increase a dimension of one-hot vector for dot production
            target_one_hot = np.expand_dims(target_one_hot, axis=1)
            hidden_emb = np.dot(self._hidden_weight.T, target_one_hot)

            context_start_idx = max(0, target_idx - window_size)
            context_end_idx = min(target_idx + window_size, len(word_set))

            score_diff = []

            # Initialize variable diff. diff is a sum of losses
            diff = np.zeros(self.n_input_dim)
            for context_idx, context_word in enumerate(word_set):
                if context_start_idx <= context_idx <= context_end_idx and context_idx != target_idx:
                    # convert a context word to one-hot vector
                    context_one_hot = np.zeros(34, dtype=np.float_)
                    context_one_hot[context_word] = 1.0

                    out_emb = np.dot(self.out_weight, hidden_emb).squeeze()
                    probs = softmax(out_emb)

                    self.result.append(probs)
                    self.result.append(context_one_hot)

                    error = cross_entropy_loss(probs, context_one_hot)
                    error_2 = np.power(context_one_hot - probs, 2)
                    # score = f1_score(np.round(context_one_hot), np.round(probs), average='macro')

                    diff += error + error_2
                    # score_diff.append(score)

            self.backward(diff, hidden_emb, target_one_hot)

            losses.append(np.sum(diff))
            score_diffs.append(np.average(score_diff))
        self.losses.append(np.average(losses))
        # score = np.average(score_diffs)
        # print(score)

    def backward(self, diff: np.array, hidden_emb: np.ndarray, target_one_hot: np.ndarray) -> None:
        temp = np.outer(hidden_emb, diff)
        EH = np.outer(target_one_hot, np.dot(self.out_weight.T, diff))

        self.out_weight += self.alpha * temp.T
        self._hidden_weight += self.alpha * EH

    def train(self, epochs: int, words: List[str], n_input_dim: int, window_size: int):
        self.n_input_dim = n_input_dim

        # convert the type of words to int from string
        word_sets = self._label_encode_word_set(words)

        # forward
        for epoch in range(epochs):
            self.losses = []
            for word_set in word_sets:
                self.forward(word_set, window_size)

            avg_loss = np.average(self.losses)
            self.final_loss.append(avg_loss)
            logging.info(f'Epoch: {epoch}, Loss: {avg_loss}')

            # early stopping
            if avg_loss > np.average(self.final_loss[-2:]):
                return self.final_loss

        return self.final_loss

    @property
    def hidden_weight(self) -> np.ndarray:
        return self._hidden_weight

    @hidden_weight.setter
    def hidden_weight(self, value):
        self._hidden_weight = value
