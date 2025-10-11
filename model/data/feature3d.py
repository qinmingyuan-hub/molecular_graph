# -*- coding: UTF-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw


def smiles_to_graph(smiles):
    # Convert SMILES to a molecule using RDKit
    mol = Chem.MolFromSmiles(smiles)

    # Create an empty graph
    old_graph = nx.Graph()

    # Add atoms as nodes
    for atom in mol.GetAtoms():
        old_graph.add_node(atom.GetSymbol() + str(atom.GetIdx()))

    # Add bonds as edges
    for bond in mol.GetBonds():
        atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        old_graph.add_edge(atom1.GetSymbol() + str(atom1.GetIdx()), atom2.GetSymbol() + str(atom2.GetIdx()))

    return old_graph


def create_new_graph(old_graph):
    # New graph
    new_graph = nx.Graph()

    # Edges as nodes in the new graph
    edges = list(old_graph.edges())
    new_nodes = [f'e{i}' for i in range(len(edges))]
    new_graph.add_nodes_from(new_nodes)

    # Record the correspondence between new graph edges and old graph nodes
    edge_mapping = {}

    # Connect adjacent edges
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if set(edges[i]) & set(edges[j]):  # If the two edges share a common node
                new_graph.add_edge(f'e{i}', f'e{j}')
                edge_mapping[(f'e{i}', f'e{j}')] = list(set(edges[i]) | set(edges[j]))

    return new_graph, edge_mapping


def main(smiles):
    old_graph = smiles_to_graph(smiles)
    new_graph, edge_mapping = create_new_graph(old_graph)

    # Print the nodes and edges of the new graph
    print("Nodes of the new graph:", new_graph.nodes())
    print("Edges of the new graph:", new_graph.edges())

    # Print the correspondence list
    print("\nCorrespondence list:")
    for edge, atoms in edge_mapping.items():
        print(edge, "->", atoms)

    return new_graph, edge_mapping

def get_mol(smiles):  # Obtain the corresponding 3D information from SMILES
    mol = AllChem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except ValueError as e:  # If optimization fails, return the unoptimized version
        print("An error occurred while optimizing the molecule:", e)
        print(smiles)
    return mol

def get_molecule_positions(suppl):
    positions = []
    mol = suppl  # For self-generated molecules, it's like this. For SDF files downloaded via CID, you need mol = suppl[0]
    if mol is not None:
        try:
            conformer = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            positions.append([list(conformer.GetAtomPosition(i)) for i in range(num_atoms)])
            return [list(conformer.GetAtomPosition(i)) for i in range(num_atoms)]
        except ValueError as e:
            print("An error occurred while obtaining molecular positions:", e)
    return positions