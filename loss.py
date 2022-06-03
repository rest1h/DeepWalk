import cupy as cp


def cross_entropy_loss(y, y_hat):
    return -cp.multiply(y_hat, cp.log(y + 1e-7))
