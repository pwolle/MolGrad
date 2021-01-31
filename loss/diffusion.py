import tensorflow as tf
from .utils import *


def sdiffusion(x, t):
    t = broadcast(t, x)

    z = tf.random.normal(x.shape)
    zs = tf.nn.sigmoid(z)

    x_tilde = interpolate(x, zs, t)

    return x_tilde, z


@tf.function
def sdiffusion_loss(x, model):
    t = tf.random.uniform([x.shape[0]], 0, 1)
    x_tilde, z = sdiffusion(x, t)

    h = model(x_tilde)
    loss = tf.abs(h + z)

    return flat_sum_batch_mean(loss)


def molecule_sdiffusion(bonds, atoms, t):

    bond_t = broadcast(t, bonds)

    bond_z = tf.random.normal(bonds.shape)
    bond_z = preprocess_bond_noise(bond_z)
    bond_zs = tf.nn.sigmoid(bond_z)

    bond_tilde = interpolate(bonds, bond_zs, bond_t)

    atom_tilde, atom_z = sdiffusion(atoms, t)

    return (bond_tilde, bond_z), (atom_tilde, atom_z)


@tf.function
def molecule_sdiffusion_loss(bonds, atoms, model):
    t = tf.random.uniform([bonds.shape[0]], 0, 1)

    (bond_tilde, bond_z), (atom_tilde, atom_z) = molecule_sdiffusion(bonds, atoms, t)

    bond_h, atom_h = model(bond_tilde, atom_tilde)

    bond_loss = flat_sum_batch_mean(tf.abs(bond_h + bond_z))
    atom_loss = flat_sum_batch_mean(tf.abs(atom_h + atom_z))

    return bond_loss + atom_loss
