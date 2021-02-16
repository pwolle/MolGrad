import tensorflow as tf
from .prepare import *


def polyethylene(
        lenght,
        batch_size=1,
        bond_types=[0, 1, 1.5, 2, 3, 4],
        atom_types=['X', 'C', 'O', 'N', 'S', 'Cl']):

    bond_dict = index_dict(bond_types)
    atom_dict = index_dict(atom_types)

    smiles_string = 'C' * lenght
    bonds, atoms = decode_smiles(smiles_string)

    bonds = relabel_bonds(bonds, bond_dict)
    atoms = relabel_atoms(atoms, atom_dict)

    bonds = tf.one_hot(bonds, len(bond_types), axis=-1)
    bonds = bonds[tf.newaxis, ..., 1:]
    bonds = tf.convert_to_tensor(bonds, tf.float32)
    bonds = tf.tile(bonds, [batch_size, 1, 1, 1])

    atoms = tf.one_hot(atoms, len(atom_types), axis=-1)
    atoms = atoms[tf.newaxis, ..., 1:]
    atoms = tf.convert_to_tensor(atoms, tf.float32)
    atoms = tf.tile(atoms, [batch_size, 1, 1])

    return bonds, atoms
