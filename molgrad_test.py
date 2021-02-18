import tensorflow as tf
import numpy as np

from model.transformer import Transformer
from data.utils import save_molecules, postprocess
from data import get_gdbs
from loss.utils import *
from math import sqrt
from pprint import pprint


num_atoms = 10

num_layers = 12
bond_depth = 256
atom_depth = 512
num_heads = 16


name = f'model_{num_atoms}_{num_layers}_{bond_depth}_{atom_depth}_{num_heads}'


dataset = get_gdbs(64, num_atoms)

model = Transformer(num_layers, bond_depth, atom_depth, num_heads)

b, a = next(iter(dataset))
model(b, a)  # to init model

model.load_weights('model/saved/gdb/' + name)


bond = tf.random.normal(b.shape)
bond = preprocess_bond_noise(bond)

atom = tf.random.normal(a.shape)


N = 100
tau = 3

lmbda = 0.5
lmbda_inv = 1 / lmbda

diverge = 1e-2

alpha_0 = lmbda


for i in range(N):
    alpha = alpha_0 * tf.exp(-tau * i / N)

    bond_s = tf.nn.sigmoid(bond)
    atom_s = tf.nn.sigmoid(atom)

    bond_grad, atom_grad = model(bond_s, atom_s)

    bond_gra = alpha * lmbda_inv * bond_grad
    atom_gra = alpha * lmbda_inv * atom_grad

    bond_z = tf.random.normal(bond.shape)
    bond_z = preprocess_bond_noise(bond_z)

    atom_z = tf.random.normal(atom.shape)

    bond_dif = sqrt(2 * alpha) * lmbda * bond_z  # * 1e-1
    atom_dif = sqrt(2 * alpha) * lmbda * atom_z  # * 1e-1

    bond_div = alpha * diverge * bond
    atom_div = alpha * diverge * atom

    bond = bond + bond_gra + bond_dif + bond_div
    atom = atom + atom_gra + atom_dif + atom_div

    if i % (N // 10) == 0:
        print(i, alpha.numpy())


bond, atom = postprocess(bond, atom)

save_molecules(bond, atom)

pprint(atom[0].numpy())
pprint(bond[0].numpy())
