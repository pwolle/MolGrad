from rdkit.Chem.QED import qed
import tensorflow as tf
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw

from tqdm import tqdm
from random import shuffle

from loss import preprocess_bond_noise, s_diffusion, invsigmoid
from data import get_logp, get_qed, get_sas


def SmilesFromGraph(node_list, adjacency_matrix):
    '''
    from https://stackoverflow.com/questions/51195392/smiles-from-graph
    (with slight modification me (@unitdeterminant)))
    by @JoshuaBox

    which is a simplification of https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
    by @dakoner
    '''

    mol = Chem.RWMol()

    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])

        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            if iy <= ix:
                continue

            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE

            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    mol = mol.GetMol()
    if mol is None:
        return False, None

    smiles = Chem.MolToSmiles(mol)

    if Chem.MolFromSmiles(smiles) is None:
        return False, None

    smiles = Chem.CanonSmiles(smiles)
    return True, smiles


def save_molecules(atoms, bonds, path='dump/molecules.npy'):
    with open(path, 'wb') as f:
        np.save(f, atoms)
        np.save(f, bonds)


def read_molecules(path='dump/molecules.npy'):
    with open(path, 'rb') as f:
        atoms = np.load(f)
        bonds = np.load(f)

    return atoms, bonds


def postprocess(atoms, bonds, atom_types=[6, 8, 7, 16, 17]):
    atoms = tf.nn.sigmoid(atoms)
    atoms = tf.math.argmax(atoms, -1).numpy()

    vc = tf.zeros([bonds.shape[0], bonds.shape[1], bonds.shape[2], 1])
    bonds = tf.concat([vc, bonds], -1)
    bonds = tf.nn.sigmoid(bonds)
    bonds = tf.math.argmax(bonds, -1).numpy()

    smiles_list = []

    for a, b in zip(atoms, bonds):
        a = [atom_types[i] for i in a]

        valid, smiles = SmilesFromGraph(a, b)

        if valid and not (('.' in smiles) or (Chem.MolFromSmiles(smiles) is None)):
            smiles_list.append(smiles)

    return smiles_list


def check_valid(smiles, unstable=['OO', 'O(O)', 'NNN', 'N(N)N', 'NN(N)']):
    mol = Chem.MolFromSmiles(smiles)
    if not mol is None:
        if not '.' in smiles:
            # naive filter for stability
            if not any((u in smiles) for u in unstable):
                return True
    return False


def check_novel(smiles, dataset):
    return not (smiles in dataset)


def check_unique(smiles, covered):
    return not (smiles in covered)


def nuvfilter(smiles_list, dataset=[], nuv=[True, True, True]):
    smiles_result = []

    for smiles in smiles_list:
        smiles = Chem.CanonSmiles(smiles)
        if check_valid(smiles) or not nuv[0]:
            if check_novel(smiles, dataset) or not nuv[2]:
                smiles_result.append(smiles)

    if nuv[1]:  # only unique smiles
        smiles_result = list(set(smiles_result))

    return smiles_result


class SAnnealedLangevin:
    """
    implements "annealed langevin dynamics in the s-function"

    """

    def __init__(self,
                 model=lambda a, b: (a, b),
                 atom_shape=[6, 5], bond_shape=[6, 6, 3],
                 N=50, tau=3, temp=0.63, eta=0.1):

        self.model = model

        self.atom_shape = atom_shape
        self.bond_shape = bond_shape

        self.N = N
        self.tau = tau
        self.temp = temp
        self.eta = eta

        self.alpha_0 = min(temp, 1 / temp)

    @tf.function
    def _langevin_step(self, atoms, bonds, i, t_add=1):
        alpha = self.alpha_0 * tf.exp(-self.tau * (i / self.N + (1 - t_add)))

        # predicted step
        atoms_grad, bonds_grad = self.model(
            tf.nn.sigmoid(atoms), tf.nn.sigmoid(bonds))

        atoms_grad = atoms_grad * (alpha / self.temp)
        bonds_grad = bonds_grad * (alpha / self.temp)

        # diffusion step
        atoms_noise = tf.random.normal(atoms.shape)

        bonds_noise = tf.random.normal(bonds.shape)
        bonds_noise = preprocess_bond_noise(bonds_noise)

        diffusion_factor = tf.math.sqrt(2 * alpha) * self.temp

        atoms_diffusion = atoms_noise * diffusion_factor
        bonds_diffusion = bonds_noise * diffusion_factor

        # divergence step
        atoms_diverge = atoms * (alpha * self.eta)
        bonds_diverge = bonds * (alpha * self.eta)

        # euler integrate
        atoms += atoms_grad + atoms_diffusion + atoms_diverge
        bonds += bonds_grad + bonds_diffusion + bonds_diverge
        return atoms, bonds

    def _run(self, atoms, bonds, t_add=1, fn=lambda a, b, *_: (a, b)):
        for i in tqdm(range(self.N)):
            i = tf.cast(i, tf.float32)
            atoms, bonds = self._langevin_step(atoms, bonds, i, t_add)
            atoms, bonds = fn(atoms, bonds, i, t_add)

        return atoms, bonds

    def _get_minimization_fn(self, energy, strength):
        @tf.function
        def fn(atoms, bonds, i, t_add):
            with tf.GradientTape() as tape:
                tape.watch([atoms, bonds])
                e = energy(atoms, bonds)

            alpha = self.alpha_0 * tf.exp(-self.tau * (i / self.N + (1 - t_add)))
            atom_grads, bond_grads = tape.gradient(e, [atoms, bonds])
            atoms = atoms - alpha * strength * atom_grads
            bonds = bonds - alpha * strength * preprocess_bond_noise(bond_grads)
            return atoms, bonds

        return fn

    def generate_random(self, batch_size):
        atoms = tf.random.normal([batch_size] + self.atom_shape)

        bonds = tf.random.normal([batch_size] + self.bond_shape)
        bonds = preprocess_bond_noise(bonds)

        return self._run(atoms, bonds)

    def edit_molecules(self, atoms, bonds, t_hat):
        atoms, _ = s_diffusion(atoms, t_hat)
        bonds, _ = s_diffusion(bonds, t_hat, preprocess_bond_noise)

        atoms, bonds = invsigmoid(atoms), invsigmoid(bonds)
        return self._run(atoms, bonds, t_hat)

    def optimize_molecules(self, atoms, bonds, t_hat, energy, strength=1.):
        atoms, _ = s_diffusion(atoms, t_hat)
        bonds, _ = s_diffusion(bonds, t_hat, preprocess_bond_noise)

        fn = self._get_minimization_fn(energy, strength)

        atoms, bonds = invsigmoid(atoms), invsigmoid(bonds)
        return self._run(atoms, bonds, t_hat, fn)

    def optimize_random(self, batch_size, energy, strength=1.):
        atoms = tf.random.normal([batch_size] + self.atom_shape)

        bonds = tf.random.normal([batch_size] + self.bond_shape)
        bonds = preprocess_bond_noise(bonds)

        fn = self._get_minimization_fn(energy, strength)

        return self._run(atoms, bonds, fn=fn)


class MolPlotter:
    def __init__(self, smiles_list):
        shuffle(smiles_list)
        self.mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        self.len = len(smiles_list)
        self.i = 0

    def __call__(self, nperrow=10, nrows=2, size=150):
        n = nperrow * nrows

        mols = self.mols[self.i:self.i + n]

        if (self.i + n) > self.len:
            self.i = 0
        else:
            self.i += n

        return Draw.MolsToGridImage(
            mols,
            molsPerRow=nperrow,
            maxMols=n,
            useSVG=True,
            subImgSize=(size, size))


def get_properties(smiles_list):
    logps = []
    qeds = []
    sass = []

    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        logps.append(get_logp(molecule))
        qeds.append(get_qed(molecule))
        sass.append(get_sas(molecule))

    return {
        'logP': np.array(logps),
        'QED': np.array(qeds),
        'SAS': np.array(sass)}


def get_failures(smiles_list, threshold=6):
    failure_sas = get_properties(smiles_list)['SAS']
    failure_smiles = []

    for sas, smiles in sorted(zip(failure_sas, smiles_list))[::-1]:
        if sas > threshold:
            failure_smiles.append(smiles)

    return failure_smiles
