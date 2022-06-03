import cupy as cp


def softmax(x: cp.array) -> cp.array:
    e_x = cp.exp(x - cp.max(x))
    return e_x / e_x.sum()
