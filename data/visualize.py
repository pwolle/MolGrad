from .prepare import *
from rdkit import Chem


def reverse_index_dict(types_list):
    return {i: k for i, k in enumerate(types_list)}


def get_rdkit_molecule(bonds, atoms,
                       bond_types=[0, 1, 1.5, 2, 3, 4],
                       atom_types=['X', 'C', 'O', 'N', 'S', 'Cl']):

    bond_dict = reverse_index_dict(bond_types)
    atom_dict = reverse_index_dict(atom_types)

    bonds = relabel_bonds(bonds, bond_dict)
    atoms = relabel_atoms(atoms, atom_dict)
    
    smiles = SmilesFromGraph(atoms, bonds)
    # print(smiles)

    mol = Chem.MolFromSmiles(smiles)
    return mol, smiles


def SmilesFromGraph(node_list, adjacency_matrix):
    ''' 
    from stackoverflow.com/questions/51195392/smiles-from-graph (slight modification)
    by @JoshuaBox 

    simplification of https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
    by @dakoner

    '''

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        try: 
            a = Chem.Atom(node_list[i])
        except:
            return 'X'

        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
            elif bond == 4:
                bond_type = Chem.rdchem.BondType.QUADRUPLE
            elif bond == 1.5:
                # I (PW) don't know have a plan to check validity for aromatic bonds
                # so they will be converted to single bonds for now
                bond_type = Chem.rdchem.BondType.SINGLE

            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    
    return smiles
