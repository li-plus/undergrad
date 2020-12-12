import numpy as np


def dp(x, weight):
    assert (len(x) == weight.shape[0])

    prev = np.zeros((weight.shape[1],), dtype=np.int32)
    y = np.zeros((weight.shape[1],))

    for i in range(weight.shape[1]):
        curr_weight = weight[:, i] * x
        max_idx = np.argmax(curr_weight)
        prev[i] = max_idx
        y[i] = curr_weight[max_idx]

    return y, prev


def viterbi(x, weights):
    pred = []
    y = x

    for weight in weights:
        y, p = dp(y, weight)
        pred.append(p)

    path = [int(np.argmax(y))]
    for p in pred[::-1]:
        path.append(p[path[-1]])
    path.reverse()

    return path


if __name__ == '__main__':
    w1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
    w2 = np.array([[1, 2, 3, 4],
                   [3, 4, 5, 6],
                   [5, 6, 7, 8]])
    w3 = np.array([[6, 5], [6, 5], [4, 3], [2, 1]])
    path = viterbi(np.array([3, 1]), [w1, w2, w3])
    print(path)
