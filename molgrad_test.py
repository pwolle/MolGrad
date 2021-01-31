import tensorflow as tf

from model.transformer import Transformer
from loss import molecule_sdiffusion_loss
from generation import langevin_step
from data import get_gdbs
from loss.utils import *
from math import sqrt
from pprint import pprint


num_atoms = 6

num_layers = 12
bond_depth = 256
atom_depth = 512
num_heads = 16


name = f'model_{num_atoms}_{num_layers}_{bond_depth}_{atom_depth}_{num_heads}'


dataset = get_gdbs(1, num_atoms)

model = Transformer(num_layers, bond_depth, atom_depth, num_heads)

b, a = next(iter(dataset))
model(b, a)  # to init model

model.load_weights('model/saved/gdb/' + name)


bond = tf.random.normal(b.shape)
bond = preprocess_bond_noise(b)

atom = tf.random.normal(a.shape)


N = 100
tau = 3

lmbda = 0.36
lmbda_inv = 1 / lmbda

diverge = 1e-2

alpha_0 = lmbda


for i in range(N):
    # bond = mask_diagonal(bond)

    alpha = alpha_0 * tf.exp(-tau * i / N)

    bond_s = tf.nn.sigmoid(bond)
    atom_s = tf.nn.sigmoid(atom)

    bond_grad, atom_grad = model(bond_s, atom_s)

    bond_gra = alpha * lmbda_inv * bond_grad
    atom_gra = alpha * lmbda_inv * atom_grad

    bond_z = tf.random.normal(bond.shape)
    bond_z = preprocess_bond_noise(bond_z)

    atom_z = tf.random.normal(atom.shape)

    bond_dif = 2 * sqrt(alpha) * lmbda * bond_z  # * 1e-1
    atom_dif = 2 * sqrt(alpha) * lmbda * atom_z  # * 1e-1

    bond_div = alpha * diverge * bond
    atom_div = alpha * diverge * atom

    bond = bond + bond_gra + bond_dif + bond_div
    atom = atom + atom_gra + atom_dif + atom_div

    if i % (N // 10) == 0:
        print(i, alpha.numpy())

        # print(tf.reduce_max(bond).numpy())
        # print(tf.reduce_max(atom).numpy())


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


bond, atom = postprocess(bond, atom)


pprint(atom[0].numpy())
pprint(bond[0].numpy())
