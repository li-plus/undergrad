from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        loss = 0.5 * np.square(input - target).sum(axis=1).mean()
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        num_samples = input.shape[0]
        output = (input - target) / num_samples
        return output


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self._saved_tensor = None

    def forward(self, input, target):
        '''Your codes here'''
        target = target.astype(np.bool)
        exp_input = np.exp(input)
        softmax_input = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        self._saved_tensor = softmax_input
        loss = -np.log(softmax_input[target])
        loss = np.mean(loss)
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        target = target.astype(np.bool)
        output = self._saved_tensor
        output[target] -= 1
        output /= input.shape[0]
        return output
