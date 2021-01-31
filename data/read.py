import tensorflow as tf
import os


def decode_example(example, **extra_features):
    features = {
        'num_atoms': tf.io.FixedLenFeature([], tf.int64),
        'bonds': tf.io.FixedLenFeature([], tf.string),
        'atoms': tf.io.FixedLenFeature([], tf.string)}

    features.update(extra_features)
    example = tf.io.parse_single_example(example, features)
    return example


def parse_molecule(features, num_bond_types=6, num_atom_types=6):
    # defualt one_hot encoding of molecuels
    num_atoms = features['num_atoms']

    bonds = tf.io.decode_raw(features['bonds'], tf.uint8)
    bonds = tf.reshape(bonds, [num_atoms, num_atoms])
    bonds = tf.one_hot(bonds, num_bond_types, axis=-1)
    bonds = bonds[..., 1:]  # leave out empty bonds

    atoms = tf.io.decode_raw(features['atoms'], tf.uint8)
    atoms = tf.reshape(atoms, [num_atoms])
    atoms = tf.one_hot(atoms, num_atom_types, axis=-1)
    atoms = atoms[..., 1:]  # leave out empty atoms

    return bonds, atoms


@tf.function
def gdb_parse(example):
    features = decode_example(example)
    return parse_molecule(features)


@tf.function
def freesolv_parse(example):
    solubility_feature = tf.io.FixedLenFeature([], tf.float32)
    features = decode_example(example, solubility=solubility_feature)
    bonds, atoms = parse_molecule(features)
    solubility = features['solubility']
    return bonds, atoms, solubility


def get_gdbs(batch_size, num_atoms, path='./data/tfrecords/gdb', split='train'):
    path = os.path.join(path, f'{num_atoms}/{split}.tfrecord')
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(gdb_parse, -1)
    dataset = dataset.batch(batch_size).cache()
    return dataset


def get_freesolv(batch_size, path='./data/tfrecords/freesolv', split='train'):
    path = os.path.join(path, f'{split}.tfrecord')
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(freesolv_parse, -1)
    dataset = dataset.batch(batch_size).cache()
    return dataset


def get_toy(batch_size):
    a00 = tf.convert_to_tensor([0, 0], tf.float32)
    a01 = tf.convert_to_tensor([0, 1], tf.float32)
    a10 = tf.convert_to_tensor([1, 0], tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices([a00, a01, a10])
    dataset = dataset.repeat(1024)  # pretend one epoch is longer
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    dataset = get_freesolv(1)

    x = next(iter(dataset))

    print(x)
