import tensorflow as tf

tl = tf.keras.layers


# TODO: add rsqrt(2) scaling option to residuals

# input-shapes:
# b: [batch_size, num_atoms, num_atoms, bond_channels]
# a: [batch_size, num_atoms, atom_channels]


def split_heads(x, num_heads):
    batch_size, num_atoms, channels = x.shape
    return tf.reshape(x, [batch_size, num_heads, num_atoms, -1])


def concat_heads(x):
    batch_size, num_heads, num_atoms, channels = x.shape
    return tf.reshape(x, [batch_size, num_atoms, -1])


def mask_diagonal(x):
    m1 = tf.ones_like(x)
    me = tf.eye(x.shape[1], batch_shape=[x.shape[0]])[..., tf.newaxis]
    m = m1 - me
    return x * m


class AddDummyAtom(tl.Layer):
    def __init__(self, atom_in=2):
        super(AddDummyAtom, self).__init__()

        self.dense = tl.Dense(atom_in)
        self.padding = [[0, 0], [0, 1], [0, 1], [0, 0]]

    def call(self, bonds, atoms):
        dummy_atom = tf.ones([1, 1, 1])
        dummy_atom = self.dense(dummy_atom)
        dummy_atom = tf.tile(dummy_atom, [atoms.shape[0], 1, 1])

        atoms = tf.concat([atoms, dummy_atom], 1)

        dummy_bonds = tf.ones([1, bonds.shape[1], bonds.shape[2], 1])
        dummy_bonds = 1. - tf.pad(dummy_bonds, self.padding)
        dummy_bonds = tf.tile(dummy_bonds, [bonds.shape[0], 1, 1, 1])

        bonds = tf.pad(bonds, self.padding)
        bonds = tf.concat([bonds, dummy_bonds], -1)

        return bonds, atoms


class RemoveDummyAtom(tl.Layer):
    # dummy bond channel stays unaffected
    # since some (probably) some computation has been done on it

    def call(self, bonds, atoms):
        bonds = bonds[:, :-1, :-1, :]
        atoms = atoms[:, :-1, :]
        return bonds, atoms


class DummyAtomRedeout(tl.Layer):
    # combined with global pooling for better integration into Transformer class

    def __init__(self, units, activation=None):
        super(DummyAtomRedeout, self).__init__()
        self.dense = tl.Dense(units, activation)

    def call(self, bonds, atoms):
        dummy_atom = atoms[:, -1, :]

        bond_pool = tf.reduce_mean(bonds, [1, 2])
        atom_pool = tf.reduce_mean(atoms, 1)
        pool = tf.concat([bond_pool, atom_pool, dummy_atom], -1)

        logit = self.dense(pool)
        return logit


class ReconnectAtoms(tl.Layer):
    def __init__(self, depth, num_heads, headm=1, concat=False):
        super(ReconnectAtoms, self).__init__()

        self.depth = depth
        self.num_heads = int(num_heads * headm)
        self.concat = concat

        self.dk = tf.math.sqrt(depth / self.num_heads)

        self.Wquerry = tl.Dense(depth)
        self.dense = tl.Dense(depth)

    def call(self, bonds, atoms):
        querry = self.Wquerry(atoms)
        querrys = split_heads(querry, self.num_heads)

        attention = tf.matmul(querrys, querrys, transpose_b=True)  # n, h, a, a
        attention = attention / self.dk

        new_bonds = tf.transpose(attention, [0, 2, 3, 1])

        if self.concat:
            new_bonds = tf.concat([bonds, new_bonds], -1)

        return self.dense(new_bonds)


class GatedAttention(tl.Layer):
    def __init__(self, depth, num_heads):
        super(GatedAttention, self).__init__()

        self.depth = depth
        self.num_heads = num_heads

        self.dk = tf.math.sqrt(depth / num_heads)

        self.Wquerry = tl.Dense(depth)
        self.Wvalue = tl.Dense(depth)
        self.Wkey = tl.Dense(depth)

        self.Wcond = tl.Dense(num_heads)

        self.dense = tl.Dense(depth)

    def call(self, bonds, atoms):
        querry = self.Wquerry(atoms)
        querrys = split_heads(querry, self.num_heads)

        value = self.Wvalue(atoms)
        values = split_heads(value, self.num_heads)

        key = self.Wkey(atoms)
        keys = split_heads(key, self.num_heads)

        cond = self.Wcond(bonds)
        cond = tf.transpose(cond, [0, 3, 1, 2])

        attention = tf.matmul(querrys, keys, transpose_b=True)

        attention = attention / self.dk + cond
        attention = tf.nn.softmax(attention, -1)

        results = tf.matmul(attention, values)
        result = concat_heads(results)

        return self.dense(result)


class FeedForwardMLP(tl.Layer):
    def __init__(self, depth, multiplier=1, activation='relu', dropout=0.1):
        super(FeedForwardMLP, self).__init__()

        self.main = tf.keras.Sequential([
            tl.Dense(int(depth * multiplier), activation),
            # tl.Dropout(0.1),
            tl.Dense(depth)])

    def call(self, x):
        return self.main(x)


class InputBlock(tl.Layer):
    def __init__(self, bond_depth, atom_depth, atom_in, center=True):

        super(InputBlock, self).__init__()

        self.center = center

        self.bond_dense = tl.Dense(bond_depth)
        self.atom_dense = tl.Dense(atom_depth)

        self.dummy = AddDummyAtom(atom_in)

    def call(self, bonds, atoms):
        bonds, atoms = self.dummy(bonds, atoms)

        if self.center:
            bonds = bonds * 2 - 1
            atoms = atoms * 2 - 1

        bonds = mask_diagonal(bonds)

        bonds = self.bond_dense(bonds)
        atoms = self.atom_dense(atoms)
        return bonds, atoms


class OutputBlock(tl.Layer):
    def __init__(self, bond_out, atom_out, activation=None):
        super(OutputBlock, self).__init__()

        self.bond_dense = tl.Dense(bond_out, activation=activation)
        self.atom_dense = tl.Dense(atom_out, activation=activation)

        self.dummy = RemoveDummyAtom()

    def call(self, bonds, atoms):
        bonds, atoms = self.dummy(bonds, atoms)
        bonds = self.bond_dense(bonds)
        atoms = self.atom_dense(atoms)
        bonds = mask_diagonal(bonds)
        return bonds, atoms


class ResNorm(tl.Layer):
    def __init__(self, style='BERT'):  # eiter 'BERT' or 'GPT' (not case sensitive)
        super(ResNorm, self).__init__()
        self.style = style
        self.norm = tl.LayerNormalization()
        # self.norm = tl.BatchNormalization()

    def call(self, x, r):
        if self.style.upper() == 'BERT':
            return self.norm(x + r)
        elif self.style.upper() == 'GPT':
            return r + self.norm(x)


class FeedForwardBlock(tl.Layer):
    def __init__(self, bond_depth, atom_depth):
        super(FeedForwardBlock, self).__init__()

        self.bond_ffn = FeedForwardMLP(bond_depth)
        self.atom_ffn = FeedForwardMLP(atom_depth)

        self.bond_res = ResNorm()
        self.atom_res = ResNorm()

    def call(self, bonds, atoms):
        bonds = self.bond_res(self.bond_ffn(bonds), bonds)
        atoms = self.atom_res(self.atom_ffn(atoms), atoms)
        return bonds, atoms


class AttentionBlock(tl.Layer):
    def __init__(self, bond_depth, atom_depth, num_heads):
        super(AttentionBlock, self).__init__()

        self.reconnect = ReconnectAtoms(bond_depth, num_heads)
        self.attention = GatedAttention(atom_depth, num_heads)

        self.bond_res = ResNorm()
        self.atom_res = ResNorm()

    def call(self, bonds, atoms):
        bonds = self.bond_res(self.reconnect(bonds, atoms), bonds)
        atoms = self.atom_res(self.attention(bonds, atoms), atoms)
        return bonds, atoms


class MainBlock(tl.Layer):
    def __init__(self, bond_depth, atom_depth, num_heads):
        super(MainBlock, self).__init__()

        self.attblock = AttentionBlock(bond_depth, atom_depth, num_heads)
        self.ffnblock = FeedForwardBlock(bond_depth, atom_depth)

    def call(self, bonds, atoms):
        bonds, atoms = self.attblock(bonds, atoms)
        bonds, atoms = self.ffnblock(bonds, atoms)
        return bonds, atoms


if __name__ == "__main__":
    atoms = tf.ones([32, 3, 2])
    bonds = tf.ones([32, 3, 3, 2])

    atoms = GatedAttention(512, 16)(bonds, atoms)

    # bonds = ReconnectAtoms(512, 16)(bonds, atoms)

    # bonds = tf.transpose(bonds[0], [2, 0, 1])

    # print(bonds.shape, atoms.shape)
