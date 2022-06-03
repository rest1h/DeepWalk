import math
from sklearn.metrics import f1_score
import logging
import cupy as cp
import numpy as np
from typing import List


class Softmax:
    def __init__(self):
        self.output = None
        self.y_true = None

    def forward(self, x: cp.array, y_true: cp.array) -> cp.array:
        self.y_true = y_true
        e_x = cp.exp(x - cp.max(x))
        self.output = e_x / e_x.sum()
        return self.output


def xavier_initialization(shape: tuple) -> cp.ndarray:
    scale = 1 / max(1., (sum(shape)) / len(shape))
    limit = math.sqrt(3.0 * scale)
    return cp.random.uniform(-limit, limit, size=shape)


def cross_entropy_loss(y, y_hat):
    return -cp.multiply(y_hat, cp.log(y + 1e-7))


class SkipGram(object):
    def __init__(self, n_input_dim: int, alpha: float, epochs: int, n_emb_dim: int, window_size: int):
        self.n_input_dim = n_input_dim
        self.alpha = alpha
        self.epochs = epochs
        self.n_emb_dim = n_emb_dim
        self.window_size = window_size

        # initialize weights
        self.hidden_weight = xavier_initialization(shape=(self.n_input_dim, self.n_emb_dim))  # (34, emb_dim)
        self.out_weight = xavier_initialization(shape=(self.n_input_dim, self.n_emb_dim))

        self.num_of_nodes = n_input_dim
        self.softmax = Softmax()

        self.losses = []
        self.final_loss = []
        self.temp = []
        self.result = []

    def _label_encode_word_set(self, word_set) -> List[List[int]]:
        # convert the type of words to int from string
        word2idx = {str(word): idx for idx, word in enumerate(range(self.num_of_nodes))}
        return [
            [word2idx[word] for word in words]
            for words in word_set
        ]

    def forward(self, word_set, window_size) -> None:
        losses = []
        score_diffs = []
        for target_idx, target_word in enumerate(word_set):
            # convert a target word to one-hot vector
            target_one_hot = cp.zeros(34, cp.float_)
            target_one_hot[target_word] = 1.0

            # increase a dimension of one-hot vector for dot production
            target_one_hot = cp.expand_dims(target_one_hot, axis=1)
            hidden_emb = cp.dot(self.hidden_weight.T, target_one_hot)

            context_start_idx = max(0, target_idx - window_size)
            context_end_idx = min(target_idx + window_size, len(word_set))

            score_diff = []

            # Initialize variable diff. diff is a sum of losses
            diff = cp.zeros(self.num_of_nodes)
            for context_idx, context_word in enumerate(word_set):
                if context_start_idx <= context_idx <= context_end_idx and context_idx != target_idx:
                    # convert a context word to one-hot vector
                    context_one_hot = cp.zeros(34, dtype=cp.float_)
                    context_one_hot[context_word] = 1.0

                    out_emb = cp.dot(self.out_weight, hidden_emb).squeeze()
                    probs = self.softmax.forward(out_emb, context_one_hot)

                    self.result.append(probs)
                    self.result.append(context_one_hot)

                    error = cross_entropy_loss(probs, context_one_hot)
                    error_2 = cp.power(context_one_hot - probs, 2)
                    score = f1_score(np.round(context_one_hot.get()), np.round(probs.get()), average='macro')

                    diff += error + error_2
                    score_diff.append(score)

            self.backward(diff, hidden_emb, target_one_hot)

            losses.append(cp.sum(diff))
            score_diffs.append(np.average(score_diff))
        self.losses.append(cp.average(losses))
        score = np.average(score_diffs)
        # print(score)

    def backward(self, diff: cp.array, hidden_emb: cp.ndarray, target_one_hot: cp.ndarray) -> None:
        temp = cp.outer(hidden_emb, diff)
        EH = cp.outer(target_one_hot, cp.dot(self.out_weight.T, diff))

        self.out_weight += self.alpha * temp.T
        self.hidden_weight += self.alpha * EH

    def train(self, epochs: int, words: , num_of_nodes: int, window_size: int):
        self.num_of_nodes = num_of_nodes

        # convert the type of words to int from string
        word_sets = self._label_encode_word_set(words)

        # forward
        for epoch in range(epochs):
            self.losses = []
            for word_set in word_sets:
                self.forward(word_set, num_of_nodes, window_size)

            avg_loss = cp.average(self.losses)
            logging.info(f'Epoch: {epoch}, Loss: {avg_loss}')

            self.final_loss.append(avg_loss)

            # early stopping
            if avg_loss > cp.average(self.final_loss[-4:]):
                return self.final_loss

        return self.final_loss
