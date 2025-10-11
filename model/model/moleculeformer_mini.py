# -*- coding: UTF-8 -*-
'''
@Project ：Analysis_NER 
@File    ：EGNN_atom_bond.py
@Author  ：Mental-Flow
@Date    ：2024/11/14 11:36 
@introduction : Variable molecular fingerprints
'''

import numpy as np
import torch
import torch.nn as nn
from model.data import GetPubChemFPs, create_bond_graph, edge_get_bond_features_dim,create_atom_graph_mini, edge_get_atom_features_dim
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import GCNConv
from model.data.egnn_pytorch_global import EGNN,EGNN_Network
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from model.data.scikit_fingerprint import *

class TransformerEncoderWithCLS(nn.Module):
    def __init__(self, args, d_model=100, nhead=4, num_layers=2, dim_feedforward=200, dropout=0.1):
        super(TransformerEncoderWithCLS, self).__init__()
        # Define a trainable [CLS] initialization token
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.cls_token = nn.Parameter(torch.randn(1, d_model))
        # Define the Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        # Define the entire Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # Concatenate the [CLS] token before the input sequence
        src_with_cls = torch.cat([self.cls_token, src], dim=0)
        # The input src_with_cls should have shape (S, N, E), where S is the sequence length, N is the batch size (here 1), and E is the feature dimension (here 100)
        output = self.transformer_encoder(src_with_cls)
        # Extract the output corresponding to [CLS]
        cls_output = output[0:1, :]
        return cls_output

class moleculeformer_mini(nn.Module):
    def __init__(self, args):
        super(moleculeformer_mini, self).__init__()
        self.args = args
        self.cuda = args.cuda
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.atom_Transformer = TransformerEncoderWithCLS(self.args)
        self.GCN_atom = GCNOne(self.args, edge_get_atom_features_dim() - 3)
        self.fpn = FPN(self.args)

        self.hidden_dim = args.hidden_size
        self.dropout = args.dropout

        if args.dataset_type == 'classification':
            self.fp_dim = 4411
        if args.dataset_type == 'regression':
            self.fp_dim = 2442

        self.graph_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fp_proj = nn.Sequential(
            nn.Linear(self.fp_dim, self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        self.Linear_out1 = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.Linear_out2 = nn.Linear(self.hidden_dim * 2, args.task_num)

        self.is_classif = self.args.dataset_type == "classification"
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def forward(self, smiles):
        atom_graph = create_atom_graph_mini(smiles, self.args, "NO")
        atoms_feature, atom_index = atom_graph.get_atom_feature() # Features and index records
        fpn_out = self.fpn(smiles)
        smiles_cls=[]
        atoms_feature = atoms_feature.to(self.device)

        for i, one in enumerate(smiles):  # Adjacency matrix
            mol = Chem.MolFromSmiles(one)
            atom2bond_list = []
            atom_edges = []
            for x, bond in enumerate(mol.GetBonds()): # Atom graph
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                atom2bond_list.append((x, (atom1, atom2))) # Record the edge number and the atoms on both sides
                atom_edges.extend([[atom1, atom2], [atom2, atom1]]) # Add to the molecular graph

            atom_edges = torch.tensor(atom_edges, dtype=torch.long).t().contiguous()
            atom_edges = atom_edges.to(self.device)
            atom_start, atom_size = atom_index[i]
            one_atom_feature = atoms_feature[atom_start:atom_start + atom_size]  # Features of the i-th molecule

            one_atom_feature_3d = one_atom_feature[:, -3:]
            one_atom_feature_scalar = one_atom_feature[:, :-3]

            # Convolve the features first
            one_atom_feature_scalar = self.GCN_atom(one_atom_feature_scalar, atom_edges)
            transformer_atom_out = self.atom_Transformer(one_atom_feature_scalar)
            smiles_cls.append(torch.cat([transformer_atom_out], axis=1))

        smiles_all_cls = torch.stack(smiles_cls, dim=1)[0]

        graph_proj = self.graph_proj(smiles_all_cls)  # (batch, hidden_dim)
        fp_proj = self.fp_proj(fpn_out)  # (batch, hidden_dim)

        output = torch.cat([graph_proj,fp_proj], axis=1)
        res = output
        output = self.Linear_out1(output) + res
        output = self.Linear_out2(output)

        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        return output

class GCNOne(nn.Module):  # GCN layer for a single molecule
    def __init__(self, args, nfeat):
        super(GCNOne, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.atom_dim = args.hidden_size
        self.initial_conv = GCNConv(nfeat, self.atom_dim)
        self.fnn = nn.Linear(in_features=nfeat, out_features=self.atom_dim, bias=True)
        self.conv1 = GCNConv(self.atom_dim, self.atom_dim)
        self.conv2 = GCNConv(self.atom_dim, self.atom_dim)
        # self.conv3 = GCNConv(self.atom_dim, self.atom_dim)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the model to the device

    def forward(self, x, edge_index):
        # If the edge index is empty, directly use the fully connected layer
        if len(edge_index) == 0:
            hidden = self.fnn(x)
            return hidden

        # 1st Conv layer
        # residual = x  # Save input features
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        # 2nd Conv layer
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        # hidden = hidden + residual  # Add residual connection
        # 3rd Conv layer
        residual = hidden  # Save intermediate features
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = hidden + residual  # Add residual connection
        return hidden

class FPN(nn.Module):  # Molecular fingerprint extraction. Input: SMILES, molecular fingerprint vector. Output: 100-dimensional vector
    def __init__(self, args):
        super(FPN, self).__init__()
        # self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout

        self.hidden_dim = args.hidden_size
        self.args = args

        if hasattr(args, 'dataset_type'):
            self.fp_type = args.dataset_type  # 'classification'
        else:
            self.fp_type = 'mixed'

        if self.fp_type == 'classification':
            self.fp_dim = 4411
        if self.fp_type == 'regression':
            self.fp_dim = 2442

        if hasattr(args, 'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None

        self.cuda = args.cuda
        self.device = torch.device('cuda' if self.cuda else 'cpu')


    def forward(self, smile):
        fp_list = []
        for i, one in enumerate(smile):
            fp = []
            mol = Chem.MolFromSmiles(one)
            if self.fp_type == 'classification':
                fp_ERG = get_ERGFingerprint([mol])
                fp_RDKit = get_RDKitFingerprint([mol])
                fp_ECFP = get_ECFPFingerprint([mol])
                fp.extend(fp_ERG[0])
                fp.extend(fp_RDKit[0])
                fp.extend(fp_ECFP[0])

            if self.fp_type == 'regression':
                fp_ECFP = get_ECFPFingerprint([mol])
                fp_ERG = get_ERGFingerprint([mol])
                fp_EState = get_EStateFingerprint([mol])
                fp.extend(fp_ECFP[0])
                fp.extend(fp_ERG[0])
                fp.extend(fp_EState[0])

            fp_list.append(fp)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:, self.fp_changebit - 1] = np.ones(fp_list[:, self.fp_changebit - 1].shape)
            fp_list.tolist()

        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()
        return fp_list


if __name__ =="__main__":
    from model.tool import set_train_argument, set_predict_argument, set_hyper_argument, set_interfp_argument, \
        set_intergraph_argument
    args = set_train_argument()
    args.cuda = False
    args.num_folds = 10
    args.epochs = 20
    # args.dataset_type='classification'
    args.dataset_type = 'regression'
    args.save_path='model_save'
    args.model_type='moleculeformer_mini'
    args.log_path='log'
    args.split_type = 'ratio_random'
    args.data_path = 'Data/MoleculeNet_dataset/bace.csv'
    args.dropout = 0
    model = moleculeformer_mini(args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}")

    model(['C.C','CC1(C)CC[C@]2(NC(=O)C(C)(F)F)CC[C@]3(C)[C@H](C(=O)C=C4[C@@]3(C)CC[C@H]3C(C)(C)C(=O)C(C#N)=C[C@]43C)[C@@H]2C1'])