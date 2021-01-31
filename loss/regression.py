import tensorflow as tf
from .utils import *


def regression_l1(bonds, atoms, y, model):
    h = model(bonds, atoms)
    loss = flat_sum_batch_mean(tf.abs(h - y))
    return loss
