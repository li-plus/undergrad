import numpy as np
import tensorflow as tf
from common import config

mean, std = config.mean, config.std

class Attack():

    def __init__(self, model, batchsize, alpha, beta, gamma, CW_kappa, use_cross_entropy_loss, **kwargs):
        self.batchsize = batchsize
        self.model = model  # pretrained vgg model used as classifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.CW_kappa = CW_kappa
        self.use_cross_entropy_loss = use_cross_entropy_loss
    
    '''Build computation graph for generating adversarial examples'''
    def generate_graph(self, pre_noise, x, gt, target=None, **kwargs):
        noise = 10 * tf.tanh(pre_noise) 
        x_noise = x + noise                 ## add perturbation and get adversarial examples
        x_clip = tf.clip_by_value(x_noise, 0, 255) 
        x_round = x_clip + tf.stop_gradient(x_clip // 1 - x_clip) ##skip computing gradient wrt to rounded results(x_round) and only calculate the gradient wrt to x_clip 
        x_norm = (x_round - mean)/(std + 1e-7)          ## normalize the image input for the classfier
        logits = self.model.build(x_norm)               

        if self.use_cross_entropy_loss:
            loss = self.cross_entropy_loss(gt, target, logits, target is not None)
        else:
            loss = self.CW_attack_loss(gt, target, logits, target is not None)
        reg = self.regularization((x_round - x) / 255)
        loss = tf.reduce_mean(self.alpha * loss + reg)

        preds = tf.nn.softmax(logits)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32), gt), tf.float32))
        return acc, loss, x_round

    def regularization(self, pertubations):
        # TODO: regularization
        reg = self.beta * tf.reduce_sum(tf.abs(pertubations), axis=[1, 2, 3]) + self.gamma * tf.reduce_sum(pertubations * pertubations, axis=[1, 2, 3])
        return reg

    def cross_entropy_loss(self, ground_truth, target, logits, is_targeted_attack):
        # TODO: loss definition for both untargeted attack and targeted attack
        if is_targeted_attack:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
        else: 
            loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth, logits=logits)
        return loss

    def CW_attack_loss(self, ground_truth, target, logits, is_targeted_attack):
        # TODO: loss definition for both untargeted attack and targeted attack
        if is_targeted_attack:
            nontarget_mask = tf.one_hot(target, 10, on_value=float('inf'), off_value=0.0)
            logits_nontarget = tf.reduce_max(logits - nontarget_mask, axis=1)
            target_mask = tf.one_hot(target, 10, on_value=0.0, off_value=float('inf'))
            logits_target = tf.reduce_max(logits - target_mask, axis=1)
            loss = tf.maximum(tf.constant(-self.CW_kappa, dtype=tf.float32), logits_nontarget - logits_target)
        else: 
            gt_mask = tf.one_hot(ground_truth, 10, on_value=0.0, off_value=float('inf'))
            logits_gt = tf.reduce_max(logits - gt_mask, axis=1)
            nongt_mask = tf.one_hot(ground_truth, 10, on_value=float('inf'), off_value=0.0)
            logits_nongt = tf.reduce_max(logits - nongt_mask, axis=1)
            loss = tf.maximum(tf.constant(-self.CW_kappa, dtype=tf.float32), logits_gt - logits_nongt)
        return loss

    '''Build a graph for evaluating the classification result of adversarial examples'''
    def evaluate(self, x, gt, **kwargs): 
        x = (x - mean)/(std + 1e-7)
        logits = self.model.build(x)
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                             tf.cast(tf.argmax(gt_one_hot, 1), dtype=tf.int32)), tf.float32))
        return acc
