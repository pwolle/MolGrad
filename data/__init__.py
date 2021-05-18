import tensorflow as tf
import numpy as np
import sys
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rdkit import Chem

from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig


def get_sas_calculator():
    # getting the SA score is not directly available in rdkit so this hack is necessary
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    return sascorer.calculateScore


get_sas = get_sas_calculator()


def get_qed(molecule):
    return Descriptors.qed(molecule)


def get_logp(molecule):
    return Crippen.MolLogP(molecule)


def get_normalized_props(molecule):
    logp = get_logp(molecule) - 0.5645155 / 0.7663688
    qed = get_qed(molecule) - 0.41825834 / 0.065860875
    sas = get_sas(molecule) - 3.304772 / 0.7816496
    return [logp, qed, sas]


def ensure_path(path):
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def index_label(input, types_list: list):
    x = np.array(input, dtype=int).copy().reshape([-1])

    for i, v in enumerate(x):
        x[i] = types_list.index(v)

    return x.reshape(np.shape(input))


def bytes_feature(value):
    l = tf.train.BytesList(value=[value])
    f = tf.train.Feature(bytes_list=l)
    return f


def performance_optimize(dataset, shuffle_buffer):
    return dataset.prefetch(-1).cache().shuffle(shuffle_buffer)


def get_gdb(num_atoms=6, path='data/raw/gdb13/{num_atoms:.0f}.smi'):
    def parse_line(line):
        smiles = str(line)[2:-3]
        smiles = Chem.CanonSmiles(smiles)
        return smiles

    dataset_path = path.format(num_atoms=num_atoms)
    dataset_lines = open(dataset_path, 'rb').readlines()
    dataset_smiles = [parse_line(line) for line in dataset_lines]

    return dataset_smiles


def dataFromSmiles(smiles,
                   atom_types=[6, 8, 7, 16, 17],
                   bond_types=[0, 1, 2, 3]):

    smiles = Chem.CanonSmiles(smiles)

    molecule = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(molecule)

    atomic_nums = [a.GetAtomicNum() for a in molecule.GetAtoms()]
    atoms = index_label(atomic_nums, atom_types)
    atoms = tf.one_hot(atoms, len(atom_types))

    adjacency = Chem.GetAdjacencyMatrix(molecule, True)
    bonds = index_label(adjacency, bond_types)
    bonds = tf.one_hot(bonds, len(bond_types))
    bonds = bonds[..., 1:]  # leave out empty bonds

    return atoms, bonds

def get_hexane(batch_size):
    hexane_smiles = 'CCCCCC'

    atoms, bonds = dataFromSmiles(hexane_smiles)
    atoms, bonds = atoms[tf.newaxis], bonds[tf.newaxis]

    hexane_atoms = tf.tile(atoms, [batch_size, 1, 1])
    hexane_bonds = tf.tile(bonds, [batch_size, 1, 1, 1])

    return hexane_atoms, hexane_bonds, hexane_smiles


class FixMoleculeDataset:
    """
    standard dataset format for gdb13 subsets of fixed molecule size

    """

    def __init__(self,
                 num_atoms=6,
                 supervised=False,
                 train_ratio=0.9,
                 atom_types=[6, 8, 7, 16, 17],
                 bond_types=[0, 1, 2, 3],
                 rewrite=False,
                 output_path='data/tfrecords/gdb/{num_atoms:.0f}/{split}.tfrecord',
                 source_path='data/raw/gdb13/{num_atoms:.0f}.smi', ):

        assert num_atoms > 2
        assert train_ratio > 0. and train_ratio < 1.

        self.num_atoms = num_atoms
        self.supervised = supervised
        self.train_ratio = train_ratio

        self.atom_types = atom_types
        self.bond_types = bond_types
        self.natom_types = len(atom_types)
        self.nbond_types = len(bond_types)

        self.output_path = output_path
        self.source_path = source_path.format(num_atoms=self.num_atoms)

        self._write(rewrite)

    def _parse_line(self, line):
        smiles = str(line)[2:-3]
        smiles = Chem.CanonSmiles(smiles)

        molecule = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(molecule)

        atomic_nums = [a.GetAtomicNum() for a in molecule.GetAtoms()]
        atoms = index_label(atomic_nums, self.atom_types)
        atoms = tf.one_hot(atoms, self.natom_types)

        atoms = atoms.numpy().astype(bool).tostring()
        atoms = bytes_feature(atoms)

        adjacency = Chem.GetAdjacencyMatrix(molecule, True)
        bonds = index_label(adjacency, self.bond_types)
        bonds = tf.one_hot(bonds, self.nbond_types)
        bonds = bonds[..., 1:]  # leave out empty bonds

        bonds = bonds.numpy().astype(bool).tostring()
        bonds = bytes_feature(bonds)

        props = get_normalized_props(molecule)
        props = tf.train.FloatList(value=props)
        props = tf.train.Feature(float_list=props)

        features_dict = {
            'atoms': atoms,
            'bonds': bonds,
            'props': props}

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        return example

    def _write(self, rewrite):
        train_path = self.output_path.format(num_atoms=self.num_atoms, split='train')
        test_path = self.output_path.format(num_atoms=self.num_atoms, split='test')

        # assume if test set exists if train set exists
        if not rewrite and os.path.exists(train_path):
            return

        lines = open(self.source_path, 'rb').readlines()
        train_lines, test_lines = train_test_split(
            lines, train_size=self.train_ratio)

        ensure_path(train_path)
        prefix = f'writing gdb-{self.num_atoms:02} train file'
        with tf.io.TFRecordWriter(train_path) as writer:
            for line in tqdm(train_lines, desc=prefix):
                example = self._parse_line(line)
                example = example.SerializeToString()
                writer.write(example)

        ensure_path(test_path)
        prefix = f'writing gdb-{self.num_atoms:02} test  file'
        with tf.io.TFRecordWriter(test_path) as writer:
            for line in tqdm(test_lines, desc=prefix):
                example = self._parse_line(line)
                example = example.SerializeToString()
                writer.write(example)

    @tf.function
    def _parse_example(self, example):
        features = {
            'atoms': tf.io.FixedLenFeature([], tf.string),
            'bonds': tf.io.FixedLenFeature([], tf.string),
            'props': tf.io.FixedLenFeature([3], tf.float32)}

        data = tf.io.parse_single_example(example, features)

        atoms = tf.io.decode_raw(data['atoms'], tf.bool)
        atoms = tf.reshape(atoms, [self.num_atoms, self.natom_types])
        atoms = tf.cast(atoms, tf.float32)

        bonds = tf.io.decode_raw(data['bonds'], tf.bool)
        bonds = tf.reshape(bonds, [self.num_atoms, self.num_atoms, self.nbond_types - 1])
        bonds = tf.cast(bonds, tf.float32)

        atoms.set_shape([self.num_atoms, self.natom_types])
        bonds.set_shape([self.num_atoms, self.num_atoms, self.nbond_types - 1])

        if not self.supervised:
            return atoms, bonds

        props = data['props']
        return atoms, bonds, props

    def get_split(self, split='train', optimize=True, shuffle_buffer=8192 * 4):
        path = self.output_path.format(num_atoms=self.num_atoms, split=split)
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(self._parse_example, -1)

        if optimize:
            dataset = performance_optimize(dataset, shuffle_buffer)
        return dataset





if __name__ == '__main__':
    dataset = FixMoleculeDataset(
        rewrite=True, supervised=True).get_split('train')

    dataset = dataset.batch(32)

    data = next(iter(dataset))

    print(data[0].shape, data[1].shape, data[2].shape)
