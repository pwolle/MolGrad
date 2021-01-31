import tensorflow as tf
from .layers import *


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, bond_depth, atom_depth, num_heads,
                 bond_out=5, atom_out=5, atom_in=5):

        super(Transformer, self).__init__()

        self.in_block = InputBlock(bond_depth, atom_depth, atom_in)

        self.blocks = []
        for i in range(num_layers):
            block = MainBlock(bond_depth, atom_depth, num_heads)
            self.blocks.append(block)

        self.out_block = OutputBlock(bond_out, atom_out)

    def call(self, bonds, atoms):
        bonds, atoms = self.in_block(bonds, atoms)

        for block in self.blocks:
            bonds, atoms = block(bonds, atoms)

        bonds, atoms = self.out_block(bonds, atoms)
        return bonds, atoms


class RegressionTransformer(tf.keras.Model):
    def __init__(self, num_layers, bond_depth, atom_depth, num_heads,
                 out_units=1, atom_in=5):

        super(RegressionTransformer, self).__init__()

        self.in_block = InputBlock(bond_depth, atom_depth, atom_in)

        self.blocks = []
        for i in range(num_layers):
            block = MainBlock(bond_depth, atom_depth, num_heads)
            self.blocks.append(block)

        self.readout = DummyAtomRedeout(out_units)

    def call(self, bonds, atoms):
        bonds, atoms = self.in_block(bonds, atoms)

        for block in self.blocks:
            bonds, atoms = block(bonds, atoms)

        logit = self.readout(bonds, atoms)
        return logit


if __name__ == "__main__":
    bonds = tf.ones([32, 6, 6, 5])
    atoms = tf.ones([32, 6, 5])

    model = Transformer(8, 512, 256, 16)

    bonds, atoms = model(bonds, atoms)

    print(bonds.shape, atoms.shape)
