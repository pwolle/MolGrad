import tensorflow as tf
from .utils import *
from .diffusion import *


@tf.function
def regression_l1(bonds, atoms, y, model):
    h = model(bonds, atoms)
    loss = flat_sum_batch_mean(tf.abs(h - y))
    return loss


@tf.function
def molecule_sdiffusion_regression(bonds, atoms, y, model):
    t = tf.random.uniform([bonds.shape[0]], 0, 1)

    (bond_tilde, bond_z), (atom_tilde, atom_z) = molecule_sdiffusion(bonds, atoms, t)

    h = model(bond_tilde, atom_tilde)

    loss = flat_sum_batch_mean(tf.abs(h - y))

    return loss
