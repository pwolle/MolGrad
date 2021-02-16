import tensorflow as tf

from model.transformer import Transformer, RegressionTransformer
from loss.utils import preprocess_bond_noise
from data.molecules import polyethylene
from data.utils import save_molecules, postprocess
from data import get_gdbs
from loss.utils import *
from math import sqrt
from loss.diffusion import molecule_sdiffusion
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


student = RegressionTransformer(4, 64, 128, 4)
student(b, a)  # to init model

student.load_weights('model/saved/solv/student6crippen')


def get_optgrad(bonds, atoms):
    with tf.GradientTape(persistent=True) as tape:  # persistent might introduce (momory) overhead
        tape.watch(bonds)
        tape.watch(atoms)
        bonds_s = tf.nn.sigmoid(bonds)
        atoms_s = tf.nn.sigmoid(atoms)
        logit = student(bonds, atoms)

    bond_grads = tape.gradient(logit, bonds)
    bond_grads = preprocess_bond_noise(bond_grads)
    # bond_grads = bond_grads / (tf.math.reduce_std(bond_grads) + 1e-3) * 1.5

    atom_grads = tape.gradient(logit, atoms)
    # atom_grads = atom_grads / (tf.math.reduce_std(atom_grads) + 1e-3)

    return bond_grads, atom_grads


def invsigmoid(x):
    return tf.math.log(x / (1 - x))


N = 100
tau = 3

t_0 = 0.85

lmbda = 0.39
lmbda_inv = 1 / lmbda

lmbda_opt = -62

diverge = 1e-3

alpha_0 = min(lmbda, lmbda_inv)


hexane = polyethylene(6, 64)
bond, atom = hexane

(bond, _), (atom, _) = molecule_sdiffusion(bond, atom, t_0)

bond = invsigmoid(bond)
atom = invsigmoid(atom)


for i in range(N):
    # bond = mask_diagonal(bond)

    alpha = alpha_0 * tf.exp(-tau * (i / N + (1 - t_0)))

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

    bond_opt, atom_opt = get_optgrad(bond, atom)
    bond_opt = lmbda_opt * alpha * bond_opt
    atom_opt = lmbda_opt * alpha * atom_opt

    bond = bond + bond_gra + bond_dif + bond_div + bond_opt
    atom = atom + atom_gra + atom_dif + atom_div + atom_opt

    if i % (N // 10) == 0:
        print(i, alpha.numpy())
        # print(tf.math.reduce_std(bond_opt/alpha).numpy())
        # print(tf.math.reduce_std(atom_opt/alpha).numpy())





bond, atom = postprocess(bond, atom)

save_molecules(bond, atom)


pprint(atom[0].numpy())
pprint(bond[0].numpy())
