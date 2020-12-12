import numpy as np
import math


def get_mse(pred, test):
    pred = np.array(pred, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    return np.mean((pred - test) ** 2)


def get_psnr(pred, test, max_val=255):
    mse = get_mse(pred, test)
    if mse == 0:
        return math.inf
    else:
        return 10 * math.log10(max_val ** 2 / get_mse(pred, test))


def get_rms(pred, test):
    return math.sqrt(get_mse(pred, test))
