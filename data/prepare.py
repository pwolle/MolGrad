import tensorflow as tf
import pysmiles as smi
import networkx as nx
import numpy as np
import os
import re

from sklearn.model_selection import train_test_split
from pprint import pprint
from tqdm import tqdm


def index_dict(types_list):
    return {k: i for i, k in enumerate(types_list)}


def decode_smiles(smiles_string):
    molecule = smi.read_smiles(smiles_string)

    bonds = nx.to_numpy_matrix(molecule, weight='order')
    bonds = np.array(bonds)

    atoms = molecule.nodes('element')
    atoms = [a[-1] for a in atoms]

    return bonds, atoms


def relabel_bonds(bonds, index_dict):
    for i, row in enumerate(bonds):
        for j, _ in enumerate(row):
            bonds[i][j] = index_dict[bonds[i][j]]
    return bonds


def relabel_atoms(atoms, index_dict):
    return np.array([index_dict[a] for a in atoms])


def readlines(path):
    return open(path, 'rb').readlines()


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def FloatFeature(value):
    value = tf.convert_to_tensor(float(value), dtype=tf.float32)
    value = tf.train.FloatList(value=[value])
    value = tf.train.Feature(float_list=value)
    return value


def encode_example(bonds, atoms, **extra_features):
    num_atoms = atoms.shape[0]
    num_atoms = tf.convert_to_tensor(num_atoms, tf.int64)
    num_atoms = tf.train.Int64List(value=[num_atoms])
    num_atoms = tf.train.Feature(int64_list=num_atoms)

    bonds = bonds.astype(np.uint8).tostring()
    bonds = tf.train.BytesList(value=[bonds])
    bonds = tf.train.Feature(bytes_list=bonds)

    atoms = atoms.astype(np.uint8).tostring()
    atoms = tf.train.BytesList(value=[atoms])
    atoms = tf.train.Feature(bytes_list=atoms)

    features_dict = {
        'num_atoms': num_atoms,
        'bonds': bonds,
        'atoms': atoms}

    features_dict.update(extra_features)
    features = tf.train.Features(feature=features_dict)

    example = tf.train.Example(features=features)
    return example


class GDBs:
    def __init__(
            self,
            num_atoms,
            data_path='./data/raw/gdb13/',
            out_path='./data/tfrecords/gdb',
            train_ratio=0.9,
            bond_types=[0, 1, 1.5, 2, 3, 4],
            atom_types=['X', 'C', 'O', 'N', 'S', 'Cl']):

        self.num_atoms = num_atoms
        self.train_ratio = train_ratio

        self.bond_dict = index_dict(bond_types)
        self.atom_dict = index_dict(atom_types)

        file_name = f'{num_atoms}.smi'
        self.data_path = os.path.join(data_path, file_name)

        self.out_path = os.path.join(out_path, f'{num_atoms}')
        ensure_path(self.out_path)

    def example_from_line(self, line):
        smiles_string = str(line)[2:-3]

        bonds, atoms = decode_smiles(smiles_string)

        bonds = relabel_bonds(bonds, self.bond_dict)
        atoms = relabel_atoms(atoms, self.atom_dict)

        example = encode_example(bonds, atoms)
        return example

    def write(self):
        lines = readlines(self.data_path)

        if self.num_atoms > 1:
            train_lines, test_lines = train_test_split(
                lines, train_size=self.train_ratio)
        else:
            # in case of num_atoms=1 there only is one example
            train_lines = lines
            test_lines = lines

        prefix = f'writing gdb-{self.num_atoms:02} train file'
        file_name = os.path.join(self.out_path, 'train.tfrecord')

        with tf.io.TFRecordWriter(file_name) as writer:
            for line in tqdm(train_lines, desc=prefix):
                example = self.example_from_line(line)
                example = example.SerializeToString()

                writer.write(example)

        prefix = f'writing gdb-{self.num_atoms:02} test  file'
        file_name = os.path.join(self.out_path, 'test.tfrecord')

        with tf.io.TFRecordWriter(file_name) as writer:
            for line in tqdm(test_lines, desc=prefix):
                example = self.example_from_line(line)
                example = example.SerializeToString()

                writer.write(example)


class FreeSolv:
    def __init__(
            self,
            num_atoms=6,
            data_path='./data/raw/freesolv.txt',
            out_path='./data/tfrecords/freesolv/',
            train_ratio=0.9,
            bond_types=[0, 1, 1.5, 2, 3, 4],
            atom_types=['X', 'C', 'O'],
            mean_norm=-2.33,
            std_norm=3.33):

        self.num_atoms = num_atoms
        self.atom_types = atom_types
        self.train_ratio = train_ratio

        self.mean_norm = mean_norm
        self.std_norm = std_norm

        self.bond_dict = index_dict(bond_types)
        self.atom_dict = index_dict(atom_types)

        self.data_path = data_path
        self.out_path = out_path
        ensure_path(out_path)

    def check_line(self, line):
        items = re.split(';', str(line))
        smiles_string, solubility = items[1], items[3]

        solubility = (float(solubility) - self.mean_norm) / self.std_norm

        bonds, atoms = decode_smiles(smiles_string)

        num_atoms = len(atoms)
        num_atoms_check = (num_atoms == self.num_atoms)

        elements_check = all([(e in self.atom_types) for e in atoms])

        return num_atoms_check, elements_check, {'bonds': bonds,
                                                 'atoms': atoms,
                                                 'solubility': solubility}

    def get_example(self, info_dict):
        bonds = info_dict['bonds']
        bonds = relabel_bonds(bonds, self.bond_dict)

        atoms = info_dict['atoms']
        atoms = relabel_atoms(atoms, self.atom_dict)

        solubility = info_dict['solubility']
        solubility = FloatFeature(solubility)

        example = encode_example(bonds, atoms, solubility=solubility)
        return example

    def write(self):
        lines = readlines(self.data_path)[3:]

        train_lines, test_lines = train_test_split(
            lines, train_size=self.train_ratio)

        file_name = os.path.join(self.out_path, 'train.tfrecord')

        N = 0

        with tf.io.TFRecordWriter(file_name) as writer:

            for line in train_lines:

                num_atoms_check, elements_check, info_dict = self.check_line(line)

                if num_atoms_check:
                    if elements_check:

                        example = self.get_example(info_dict)
                        example = example.SerializeToString()

                        writer.write(example)

                        N += 1

        print(N)

        file_name = os.path.join(self.out_path, 'test.tfrecord')

        with tf.io.TFRecordWriter(file_name) as writer:

            for line in test_lines:

                num_atoms_check, elements_check, info_dict = self.check_line(line)

                if num_atoms_check:
                    if elements_check:

                        example = self.get_example(info_dict)
                        example = example.SerializeToString()

                        writer.write(example)


if __name__ == '__main__':
    ds = FreeSolv()
    ds.write()

    # for i in range(1, 11):
    #     gdb = GDBs(i)
    #     gdb.write()
