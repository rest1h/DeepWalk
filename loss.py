import numpy as np


def cross_entropy_loss(y, y_hat):
    return -np.multiply(y_hat, np.log(y + 1e-7))
