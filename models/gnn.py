import tensorflow as tf
tl = tf.keras.layers


def split_heads(x, num_heads):
    batch_size, num_atoms, _ = x.shape
    return tf.reshape(x, [batch_size, num_heads, num_atoms, -1])


def concat_heads(x):
    batch_size, _, num_atoms, _ = x.shape
    return tf.reshape(x, [batch_size, num_atoms, -1])


def mask_diagonal(x):
    m1 = tf.ones_like(x)
    me = tf.eye(x.shape[1], batch_shape=[x.shape[0]])[..., tf.newaxis]
    m = m1 - me
    return x * m


def rem_dummyatom(atoms, bonds):
    atoms = atoms[:, :-1, :]
    bonds = bonds[:, :-1, :-1, :]
    return atoms, bonds


class AddDummyAtom(tl.Layer):
    def __init__(self, atom_in=5):
        super(AddDummyAtom, self).__init__()

        self.dummy_atom = self.add_weight(
            'dummy_atom', [1, 1, atom_in], tf.float32,
            tf.keras.initializers.Zeros())

        self.padding = [[0, 0], [0, 1], [0, 1], [0, 0]]

    def call(self, atoms, bonds):

        dummy_atom = tf.tile(self.dummy_atom, [atoms.shape[0], 1, 1])
        atoms = tf.concat([atoms, dummy_atom], 1)

        dummy_bonds = tf.ones([1, bonds.shape[1], bonds.shape[2], 1])
        dummy_bonds = 1. - tf.pad(dummy_bonds, self.padding)
        dummy_bonds = tf.tile(dummy_bonds, [bonds.shape[0], 1, 1, 1])

        bonds = tf.pad(bonds, self.padding)
        bonds = tf.concat([bonds, dummy_bonds], -1)
        return atoms, bonds


class DummyAtomReadout(tl.Layer):
    def __init__(self, units, act=None):
        super(DummyAtomReadout, self).__init__()
        self.affine_out = tl.Dense(units, act, name='affine_out')

    def __call__(self, atoms, bonds):
        dummy_features = atoms[:, -1, :]

        atom_features = tf.reduce_mean(atoms, 1)
        bond_features = tf.reduce_mean(bonds, [1, 2])

        features = tf.concat([dummy_features, atom_features, bond_features], -1)
        return self.affine_out(features)


class FeedForwardMLP(tl.Layer):
    def __init__(self, depth=512, num_layers=1, multiplier=1, activation='relu', name=None):
        super(FeedForwardMLP, self).__init__(name=name)

        layers = [tl.Dense(int(depth * multiplier), activation) for _ in range(num_layers)]
        layers.append(tl.Dense(depth))

        self.main = tf.keras.Sequential(layers)

    def call(self, x):
        return self.main(x)


class GatedAttention(tl.Layer):
    def __init__(self, depth=512, num_heads=16, cond_depthm=16):
        super(GatedAttention, self).__init__()

        self.depth = depth
        self.num_heads = num_heads
        self.dk = tf.math.sqrt(depth / num_heads)

        self.Wquerry = tl.Dense(depth, use_bias=False, name="Wquerry")
        self.Wvalue = tl.Dense(depth, use_bias=False, name="Wvalue")
        self.Wkey = tl.Dense(depth, use_bias=False, name="Wkey")

        self.Wcond = tl.Dense(num_heads, use_bias=False, name="Wcond")

        self.affine_out = tl.Dense(depth, name="affine_out")

    def call(self, atoms, bonds):
        querry = self.Wquerry(atoms)
        querrys = split_heads(querry, self.num_heads)

        value = self.Wvalue(atoms)
        values = split_heads(value, self.num_heads)

        key = self.Wkey(atoms)
        keys = split_heads(key, self.num_heads)

        graph_features = bonds
        cond = self.Wcond(graph_features)
        cond = tf.transpose(cond, [0, 3, 1, 2])

        attention = tf.matmul(querrys, keys, transpose_b=True)

        attention = attention / self.dk + cond
        attention = tf.nn.softmax(attention, -1)

        results = tf.matmul(attention, values)
        result = concat_heads(results)

        return self.affine_out(result)


class ReconnectAtoms(tl.Layer):
    def __init__(self, depth=256, num_heads=16, headm=4):
        super(ReconnectAtoms, self).__init__()

        self.depth = depth
        self.num_heads = int(num_heads * headm)
        self.dk = tf.math.sqrt(depth / self.num_heads)

        self.Wquerry = tl.Dense(depth, use_bias=False, name="Wquerry")
        self.affine_out = tl.Dense(depth, name='affine_out')

    def call(self, atoms):
        querry = self.Wquerry(atoms)
        querrys = split_heads(querry, self.num_heads)

        attention = tf.matmul(querrys, querrys, transpose_b=True)
        attention = attention / self.dk
        attention = tf.transpose(attention, [0, 2, 3, 1])

        return self.affine_out(attention)


class ResNorm(tl.Layer):
    def __init__(self, name=None):
        super(ResNorm, self).__init__(name=name)
        self.norm = tl.LayerNormalization()

    def call(self, x, r):
        return self.norm(x + r)


class AttentionBlock(tl.Layer):
    def __init__(self, atom_depth=512, bond_depth=256, num_heads=16):
        super(AttentionBlock, self).__init__()

        self.attention = GatedAttention(atom_depth, num_heads)
        self.reconnect = ReconnectAtoms(bond_depth, num_heads)

        self.atom_res = ResNorm(name='atom_res')
        self.bond_res = ResNorm(name='bond_res')

    def call(self, atoms, bonds):
        atoms = self.atom_res(self.attention(atoms, bonds), atoms)
        bonds = self.bond_res(self.reconnect(atoms), bonds)
        return atoms, bonds


class FeedForwardBlock(tl.Layer):
    def __init__(self, atom_depth=512, bond_depth=256):
        super(FeedForwardBlock, self).__init__()

        self.atom_ffn = FeedForwardMLP(atom_depth, name='atom_ffn')
        self.bond_ffn = FeedForwardMLP(bond_depth, name='bond_ffn')

        self.atom_res = ResNorm(name='atom_res')
        self.bond_res = ResNorm(name='bond_res')

    def call(self, atoms, bonds):
        atoms = self.atom_res(self.atom_ffn(atoms), atoms)
        bonds = self.bond_res(self.bond_ffn(bonds), bonds)
        return atoms, bonds


class MainBlock(tl.Layer):
    def __init__(self, atom_depth=512, bond_depth=256, num_heads=16):
        super(MainBlock, self).__init__()

        self.attblock = AttentionBlock(atom_depth, bond_depth, num_heads)
        self.ffnblock = FeedForwardBlock(atom_depth, bond_depth)

    def call(self, atoms, bonds):
        atoms, bonds = self.attblock(atoms, bonds)
        atoms, bonds = self.ffnblock(atoms, bonds)
        return atoms, bonds


class InputBlock(tl.Layer):
    def __init__(self, atom_depth=512, bond_depth=256, atom_in=5, cond=False, center=True):
        super(InputBlock, self).__init__()

        self.cond = cond
        self.center = center

        self.atom_dense = tl.Dense(atom_depth, name='atom_dense')
        self.bond_dense = tl.Dense(bond_depth, name='bond_dense')

        self.dummy = AddDummyAtom(atom_in)

    def call(self, atoms, bonds):
        if not self.cond:
            bonds = mask_diagonal(bonds)

        if self.center:
            bonds = bonds * 2 - 1
            atoms = atoms * 2 - 1

        atoms, bonds = self.dummy(atoms, bonds)

        atoms = self.atom_dense(atoms)
        bonds = self.bond_dense(bonds)
        return atoms, bonds


class OutputBlock(tl.Layer):
    def __init__(self, atom_depth, bond_depth, atom_out=5, bond_out=1, act=None):
        super(OutputBlock, self).__init__()

        self.atom_ffn = FeedForwardMLP(atom_depth, 2, name='atom_ffn_out')
        self.atom_dense = tl.Dense(atom_out, activation=act, name='atom_affine_out')

        self.bond_ffn = FeedForwardMLP(bond_depth, 2, name='bond_ffn_out')
        self.bond_dense = tl.Dense(bond_out, activation=act, name='bond_affine_out')

    def call(self, atoms, bonds):
        atoms, bonds = rem_dummyatom(atoms, bonds)

        bonds = self.bond_ffn(bonds)
        bonds = self.bond_dense(bonds)

        atoms = self.atom_ffn(atoms)
        atoms = self.atom_dense(atoms)

        return atoms, mask_diagonal(bonds)


class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers=6, atom_depth=512, bond_depth=256, num_heads=16,
                 atom_out=5, bond_out=3, act_out=None, cond=False):

        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.atom_depth = atom_depth
        self.bond_depth = bond_depth
        self.num_heads = num_heads
        self.atom_out = atom_out
        self.bond_out = bond_out
        self.act_out = act_out
        self.cond = cond

        self.in_block = InputBlock(atom_depth, bond_depth, atom_out, cond)

        self.blocks = []
        for i in range(num_layers):
            block = MainBlock(atom_depth, bond_depth, num_heads)
            self.blocks.append(block)

        self.out_block = OutputBlock(atom_depth, bond_depth, atom_out, bond_out, act_out)

        # to build all the layers
        self(tf.zeros([1, 2, atom_out]), tf.zeros([1, 2, 2, bond_out]))

    def call(self, atoms, bonds):
        atoms, bonds = self.in_block(atoms, bonds)

        atom_acts = [atoms]
        bond_acts = [bonds]

        for block in self.blocks:
            atoms, bonds = block(atoms, bonds)

            atom_acts.append(atoms)
            bond_acts.append(bonds)

        atoms = tf.concat(atom_acts, -1)
        bonds = tf.concat(bond_acts, -1)

        atoms, bonds = self.out_block(atoms, bonds)
        return atoms, bonds

    def __str__(self):
        name = 'g2g'
        params_1 = f'{self.num_layers}l{self.atom_depth}ad'
        params_2 = f'{self.bond_depth}bd{self.num_heads}nh'
        cond = 'cond' if self.cond else ''
        return f'{name}{params_1}{params_2}{cond}'


class RegressionTransformer(tf.keras.Model):
    def __init__(self,
                 units_out=3, act_out=None,
                 num_layers=6, atom_depth=512, bond_depth=256,
                 num_heads=16, atom_out=5, bond_out=3, cond=False):

        super(RegressionTransformer, self).__init__()

        self.units_out = units_out
        self.act_out = act_out
        self.num_layers = num_layers
        self.atom_depth = atom_depth
        self.bond_depth = bond_depth
        self.num_heads = num_heads
        self.atom_out = atom_out
        self.bond_out = bond_out
        self.cond = cond

        self.in_block = InputBlock(atom_depth, bond_depth, atom_out, cond, center=False)

        self.blocks = []
        for i in range(num_layers):
            block = MainBlock(atom_depth, bond_depth, num_heads)
            self.blocks.append(block)

        self.readout = DummyAtomReadout(units_out, act_out)

        # to build all the layers
        self(tf.zeros([1, 2, atom_out]), tf.zeros([1, 2, 2, bond_out]))

    def call(self, atoms, bonds):
        atoms, bonds = self.in_block(atoms, bonds)

        atom_acts = [atoms]
        bond_acts = [bonds]

        for block in self.blocks:
            atoms, bonds = block(atoms, bonds)

            atom_acts.append(atoms)
            bond_acts.append(bonds)

        atoms = tf.concat(atom_acts, -1)
        bonds = tf.concat(bond_acts, -1)

        logits = self.readout(atoms, bonds)
        return logits

    def __str__(self):
        name = 'g2v'
        params_1 = f'{self.num_layers}l{self.atom_depth}ad{self.bond_depth}bd'
        params_2 = f'{self.num_heads}nh{self.units_out}uo'
        cond = 'cond' if self.cond else ''
        return f'{name}{params_1}{params_2}{cond}'


if __name__ == '__main__':
    from pprint import pprint

    atoms = tf.random.normal([32, 6, 5])
    bonds = tf.random.normal([32, 6, 6, 3])

    # model = Transformer()
    # print(model)

    # a, b = model(atoms, bonds)

    # assert atoms.shape == a.shape
    # assert bonds.shape == b.shape

    # model.summary()

    # pprint([var.name for var in model.trainable_variables])

    model = RegressionTransformer()

    l = model(atoms, bonds)
    print(l.shape)
