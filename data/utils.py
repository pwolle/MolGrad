import tensorflow as tf
import numpy as np
import os


def save_molecules(bonds, atoms, path='molecules.npy'):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'wb') as f:
        np.save(f, bonds.numpy())
        np.save(f, atoms.numpy())


def read_molecules(path='molecules.npy'):
    with open(path, 'rb') as f:
        bonds = np.load(f)
        atoms = np.load(f)
    return bonds, atoms


def postprocess(v, e):
    vc = tf.zeros([v.shape[0], v.shape[1], v.shape[2], 1])
    v = tf.concat([vc, v], -1)
    v = tf.nn.sigmoid(v)
    v = tf.math.argmax(v, -1)

    ev = tf.zeros([e.shape[0], e.shape[1], 1])
    e = tf.concat([ev, e], -1)
    e = tf.nn.sigmoid(e)
    e = tf.math.argmax(e, -1)
    return v, e
