from argparse import Namespace
from rdkit import Chem
import torch
import numpy as np
from rdkit.Chem import AllChem, rdMolDescriptors
from itertools import combinations_with_replacement

# Define element list
elements = ['C', 'O', 'S', 'N', 'H']  # Add more elements as needed

# Generate all possible element combinations
element_combinations = [comb for comb in combinations_with_replacement(elements, 2) if comb != ('H', 'H')]
element_combinations.extend(
    [('C', 'P'), ('C', 'F'), ('C', 'Cl'), ('C', 'Br'), ('C', 'I'), ('C', 'Na'), ('C', 'K'), ('C', 'Mg'), ('C', 'B'),
     ('C', 'Se')])
# Create a dictionary to map combinations to labels
combination_to_label = {comb: idx for idx, comb in enumerate(element_combinations)}

bond_type_max = 100
bond_f_dim = 37
# Define bond features
bond_features_define = {
    'bond_type': [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC],  # Bond types
    'bond_bond_type': [i for i in range(len(combination_to_label))],  # Element types of the two atoms in the bond
    'bond_IsInRing': [True, False],
}

atom_bond_smile_changed = {}
atom_type_max = 100
atom_f_dim = 136
# Define atom features
atom_features_define = {
    'atom_symbol': list(range(atom_type_max)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],  # Note: Likely a typo for "chiral_type" (chiral tag)
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], }


def edge_get_bond_features_dim():
    return bond_f_dim


def edge_get_atom_features_dim():
    return atom_f_dim


def onek_encoding_unk(key, length):
    encoding = [0] * (len(length) + 1)
    index = length.index(key) if key in length else -1
    encoding[index] = 1
    return encoding


def get_bond_feature(mol, bond, type=False):
    if type == False:
        feature = onek_encoding_unk(bond.GetBondType(), bond_features_define['bond_type']) + \
                  onek_encoding_unk(GetBondAtomLable(bond), bond_features_define['bond_bond_type']) + \
                  onek_encoding_unk(bond.IsInRing(), bond_features_define['bond_IsInRing']) + \
                  [0, 0, 0] + \
                  [0]
    if type == True:
        feature = onek_encoding_unk(bond.GetBondType(), bond_features_define['bond_type']) + \
                  onek_encoding_unk(GetBondAtomLable(bond), bond_features_define['bond_bond_type']) + \
                  onek_encoding_unk(bond.IsInRing(), bond_features_define['bond_IsInRing']) + \
                  GetDirection(mol, bond.GetIdx()) + \
                  [GetBondLength(mol, bond.GetIdx())]
    return feature


def get_atom_feature(mol, atom, type=False):
    if type == False:
        feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
                  onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
                  onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
                  onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
                  onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
                  onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
                  [1 if atom.GetIsAromatic() else 0] + \
                  [atom.GetMass() * 0.01] + \
                  [0, 0, 0]

    if type == True:
        feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
                  onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
                  onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
                  onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
                  onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
                  onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
                  [1 if atom.GetIsAromatic() else 0] + \
                  [atom.GetMass() * 0.01] + \
                  list(GetAtom_3d(mol, atom.GetIdx()))
        # print(GetAtom_3d(mol, atom.GetIdx()))
    return feature


def GetBondAtomLable(bond):
    atom1 = bond.GetBeginAtom().GetSymbol()
    atom2 = bond.GetEndAtom().GetSymbol()
    bond_types = tuple(sorted([atom1, atom2]))  # Sort to ensure consistent order (e.g., (C,O) not (O,C))
    return combination_to_label.get(bond_types, -1)


def Get3dPosition(mol, type="NO"):
    try:
        mol = Chem.AddHs(mol)  # Add hydrogen atoms
        AllChem.EmbedMolecule(mol)  # Generate 3D conformation
        if type == "NO":
            return mol
        if type == "UFF":
            AllChem.UFFOptimizeMolecule(mol)  # Universal Force Field optimization
            return mol
        if type == "MMFF":
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s')  # MMFF force field optimization
            return mol
    except Exception as e:
        print(f"Error generating 3D position for molecule: {Chem.MolToSmiles(mol)}")
        return None  # Return None to indicate failure


def GetAtom_3d(mol, atom_index):
    if mol is None or mol.GetNumConformers() == 0:
        return [0, 0, 0]
    pos = mol.GetConformer().GetAtomPosition(atom_index)
    return [pos.x, pos.y, pos.z]  # Return coordinates as a list (fixed original missing conversion)


def GetDirection(mol, bond_index):
    if mol is None or mol.GetNumConformers() == 0:  # Check if molecule is valid and has a conformation
        return [0, 0, 0]

    bond = mol.GetBondWithIdx(bond_index)  # Get the bond by index

    # Get the two atoms of the bond
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    # Get the 3D coordinates of the atoms
    pos1 = mol.GetConformer().GetAtomPosition(atom1.GetIdx())
    pos2 = mol.GetConformer().GetAtomPosition(atom2.GetIdx())

    # Calculate the direction vector of the bond
    bond_direction = np.array([pos2.x, pos2.y, pos2.z]) - np.array([pos1.x, pos1.y, pos1.z])

    # Normalize the direction vector
    bond_direction_normalized = bond_direction / np.linalg.norm(bond_direction)
    return list(bond_direction_normalized)


def GetBondLength(mol, bond_index):
    if mol is None or mol.GetNumConformers() == 0:  # Check if molecule is valid and has a conformation
        return 0

    bond = mol.GetBondWithIdx(bond_index)  # Get the bond by index

    # Get the two atoms of the bond
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    # Get the 3D coordinates of the atoms
    pos1 = mol.GetConformer().GetAtomPosition(atom1.GetIdx())
    pos2 = mol.GetConformer().GetAtomPosition(atom2.GetIdx())

    # Calculate the length of the bond
    bond_length = np.linalg.norm(np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z]))
    return bond_length


def GetBondVector(mol, bond_index):
    if mol is None or mol.GetNumConformers() == 0:  # Check if molecule is valid and has a conformation
        return np.array([0, 0, 0])

    bond = mol.GetBondWithIdx(bond_index)  # Get the specified bond by index

    # Get the two atoms of the bond
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    # Get the 3D coordinates of the atoms
    pos1 = mol.GetConformer().GetAtomPosition(atom1.GetIdx())
    pos2 = mol.GetConformer().GetAtomPosition(atom2.GetIdx())

    # Calculate the bond vector (midpoint of the two atoms)
    # bond_vector = np.array([pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z])  # Original comment: Direct bond vector
    bond_vector = np.array(
        [(pos2.x + pos1.x) / 2, (pos2.y + pos1.y) / 2, (pos2.z + pos1.z) / 2])  # Bond midpoint position
    return bond_vector


class BondGraphOne:
    def __init__(self, smile, mol, bond_num, flag, args):
        self.smile = smile
        self.bond_feature = []
        self.bond_num = bond_num  # Before adding hydrogen atoms, so hydrogen bonds are not included
        for i, bond in enumerate(mol.GetBonds()):  # Traverse bonds directly; alignment requires algorithm control
            self.bond_feature.append(get_bond_feature(mol, bond, flag))
        self.bond_feature = [self.bond_feature[i] for i in range(self.bond_num)]  # Ensure consistent length


class AtomGraphOne:
    def __init__(self, smile, mol, atom_num, flag, args):
        self.smile = smile
        self.atom_feature = []
        self.atom_num = atom_num
        for i, atom in enumerate(mol.GetAtoms()):  # Traverse atoms directly; alignment requires algorithm control
            self.atom_feature.append(get_atom_feature(mol, atom, flag))
        self.atom_feature = [self.atom_feature[i] for i in range(self.atom_num)]  # Ensure consistent length


class BondGraphBatch:
    def __init__(self, graphs, args):
        smile_list = []
        for graph in graphs:
            smile_list.append(graph.smile)
        self.smile_list = smile_list
        self.smile_num = len(self.smile_list)
        self.bond_feature_dim = edge_get_bond_features_dim()
        self.bond_no = 0  # Start position (molecular level)
        self.bond_index = []  # Length of bond features for each molecule

        bond_feature = []
        for graph in graphs:  # Iterate over BondGraphOne instances
            bond_feature.extend(graph.bond_feature)
            self.bond_index.append((self.bond_no, graph.bond_num))
            self.bond_no += graph.bond_num

        self.bond_feature = torch.FloatTensor(bond_feature)

    def get_bond_feature(self):  # Return features of each bond and a list of index tuples (e.g., [(1, 25), (26, 40)])
        return self.bond_feature, self.bond_index


class AtomGraphBatch:
    def __init__(self, graphs, args):
        smile_list = []
        for graph in graphs:
            smile_list.append(graph.smile)
        self.smile_list = smile_list
        self.smile_num = len(self.smile_list)
        self.atom_feature_dim = edge_get_atom_features_dim()
        self.atom_no = 0  # Start position (molecular level)
        self.atom_index = []  # Length of atom features for each molecule

        atom_feature = []
        for graph in graphs:  # Iterate over AtomGraphOne instances
            atom_feature.extend(graph.atom_feature)
            self.atom_index.append((self.atom_no, graph.atom_num))
            self.atom_no += graph.atom_num

        self.atom_feature = torch.FloatTensor(atom_feature)

    def get_atom_feature(self):  # Return features of each atom and a list of index tuples (e.g., [(1, 25), (26, 40)])
        return self.atom_feature, self.atom_index


def create_bond_graph(smiles, mol, args):
    graphs = []
    for one in smiles:
        if one in bond_smile_changed:
            graph = bond_smile_changed[one]
        else:
            graph = BondGraphOne(one, mol, args)
            bond_smile_changed[one] = graph
        graphs.append(graph)
    return BondGraphBatch(graphs, args)


def create_atom_graph_mini(smiles, mol, args):
    graphs = []
    for one in smiles:
        if one in atom_smile_changed:
            graph = atom_smile_changed[one]
        else:
            mol = Chem.MolFromSmiles(one)
            atom_num = mol.GetNumAtoms()
            graph = AtomGraphOne(one, mol, atom_num, False, args)
            atom_smile_changed[one] = graph
        graphs.append(graph)
    return AtomGraphBatch(graphs, args)


atom_smile_changed = {}
bond_smile_changed = {}


def create_atom_bond_graph(smiles, args, force_field):
    atom_graphs = []
    bond_graphs = []
    for one in smiles:
        # Check cache first (use one cache key for both atom and bond graphs)
        if one in atom_smile_changed and one in bond_smile_changed:
            atom_graph = atom_smile_changed[one]
            bond_graph = bond_smile_changed[one]
        else:
            mol = Chem.MolFromSmiles(one)
            atom_num = mol.GetNumAtoms()
            bond_num = mol.GetNumBonds()  # Before adding hydrogen atoms, so hydrogen bonds are not included
            mol_3d = Get3dPosition(mol, type=force_field)
            flag = mol_3d is not None  # Check if 3D conformation was generated successfully
            if flag:
                mol = mol_3d  # Use the 3D-optimized molecule if available
            atom_graph = AtomGraphOne(one, mol, atom_num, flag, args)
            bond_graph = BondGraphOne(one, mol, bond_num, flag, args)
            # Cache the generated graphs
            atom_smile_changed[one] = atom_graph
            bond_smile_changed[one] = bond_graph

        atom_graphs.append(atom_graph)
        bond_graphs.append(bond_graph)
    return AtomGraphBatch(atom_graphs, args), BondGraphBatch(bond_graphs, args)


if __name__ == "__main__":
    # Example usage (commented out)
    # bondgraph = create_bond_graph(['[S+](CCCNC(=O)c1nc(sc1)-c1nc(sc1)CCNC(=O)[C@@H](NC(=O)[C@H]([C@H](O)[C@H](NC(=O)[C@@H](NC(=O)c1nc(nc(N)c1C)[C@@H]([NH2+]C[C@H]([NH3+])C(=O)N)CC(=O)N)[C@@H](O[C@@H]1O[C@@H](CO)[C@@H](O)[C@H](O)[C@@H]1O[C@H]1O[C@H](CO)[C@@H](O)[C@H](OC(=O)N)[C@@H]1O)c1nc[nH]c1)C)C)[C@H](O)C)(C)C','Fc1c2c(ccc1)[C@@]([NH+]=C2N)(C=1C=C(C)C(=O)N(C=1)CC)c1cc(ccc1)-c1cc(cnc1)C#CC'],[])
    # atomgraph = create_atom_graph(['[S+](CCCNC(=O)c1nc(sc1)-c1nc(sc1)CCNC(=O)[C@@H](NC(=O)[C@H]([C@H](O)[C@H](NC(=O)[C@@H](NC(=O)c1nc(nc(N)c1C)[C@@H]([NH2+]C[C@H]([NH3+])C(=O)N)CC(=O)N)[C@@H](O[C@@H]1O[C@@H](CO)[C@@H](O)[C@H](O)[C@@H]1O[C@H]1O[C@H](CO)[C@@H](O)[C@H](OC(=O)N)[C@@H]1O)c1nc[nH]c1)C)C)[C@H](O)C)(C)C','Fc1c2c(ccc1)[C@@]([NH+]=C2N)(C=1C=C(C)C(=O)N(C=1)CC)c1cc(ccc1)-c1cc(cnc1)C#CC'],[])
    # bondgraph = create_bond_graph(['C','[H]OC([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])(O[H])C([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)[O-])C([H])([H])c1c([H])c([H])c([H])c([H])c1[H])C([H])([H])C([H])([H])C(=O)[O-])C([H])([H])[H])C([H])([H])[H])C([H])([H])C(=O)N([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])([H])C([H])([H])C(=O)[O-])N([H])C(=O)C([H])(N([H])C(=O)C([H])(C([H])([H])c1c([H])n([H])c2c([H])c([H])c([H])c([H])c12)[N+]([H])([H])[H])C([H])([H])c1c([H])n([H])c2c([H])c([H])c([H])c([H])c12'],[])
    # atomgraph = create_atom_graph(['C','[H]OC([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])(O[H])C([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)[O-])C([H])([H])c1c([H])c([H])c([H])c([H])c1[H])C([H])([H])C([H])([H])C(=O)[O-])C([H])([H])[H])C([H])([H])[H])C([H])([H])C(=O)N([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])([H])C([H])([H])C(=O)[O-])N([H])C(=O)C([H])(N([H])C(=O)C([H])(C([H])([H])c1c([H])n([H])c2c([H])c([H])c([H])c([H])c12)[N+]([H])([H])[H])C([H])([H])c1c([H])n([H])c2c([H])c([H])c([H])c([H])c12'],[])
    create_atom_bond_graph(['C',
                            '[H]OC([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])(O[H])C([H])([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)N([H])C([H])(C(=O)[O-])C([H])([H])c1c([H])c([H])c([H])c([H])c1[H])C([H])([H])C([H])([H])C(=O)[O-])C([H])([H])[H])C([H])([H])[H])C([H])([H])C(=O)N([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H])C([H])([H])C([H])([H])C(=O)[O-])N([H])C(=O)C([H])(N([H])C(=O)C([H])(C([H])([H])c1c([H])n([H])c2c([H])c([H])c([H])c([H])c12)[N+]([H])([H])[H])C([H])([H])c1c([H])n([H])c2c([H])c([H])c([H])c([H])c12'],
                           [], 'NO')

    # Print statements (commented out for example)
    # print(bondgraph.get_bond_feature())
    # print(atomgraph.get_atom_feature())