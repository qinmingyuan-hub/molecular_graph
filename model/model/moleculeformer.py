# -*- coding: UTF-8 -*-
'''
@Project ：Analysis_NER 
@File    ：EGNN_atom_bond.py
@Author  ：Mental-Flow
@Date    ：2024/11/14 11:36 
@introduction : Variable Molecular Fingerprint
'''

import numpy as np
import torch
import torch.nn as nn
from model.data import GetPubChemFPs, create_bond_graph, edge_get_bond_features_dim, edge_get_atom_features_dim,create_atom_bond_graph
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import GCNConv
from model.data.egnn_pytorch_global import EGNN, EGNN_Network
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from model.data.scikit_fingerprint import *

# bond_f_dim = 37
# atom_f_dim = 136
class TransformerEncoderWithCLS(nn.Module):
    def __init__(self, args, d_model=100, nhead=4, num_layers=2, dim_feedforward=200, dropout=0.1):
        super(TransformerEncoderWithCLS, self).__init__()
        # Define trainable [CLS] token
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.cls_token = nn.Parameter(torch.randn(1, d_model)).to(self.device)
        # Define Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout).to(self.device)
        # Define the entire Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

    def forward(self, src):
        # Concatenate [CLS] token to the input sequence
        src_with_cls = torch.cat([self.cls_token, src], dim=0)
        # The shape of src_with_cls should be (S, N, E), where S is the sequence length, N is the batch size (here 1), and E is the feature dimension (here 100)
        output = self.transformer_encoder(src_with_cls)
        # Extract the output corresponding to [CLS]
        cls_output = output[0:1, :]
        return cls_output

class moleculeformer(nn.Module):
    def __init__(self, args):
        super(moleculeformer, self).__init__()
        self.args = args
        self.cuda = args.cuda
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.atom_3d_net = EGNN_3d_net(self.args)
        self.bond_3d_net = TransformerEncoderWithCLS(self.args)
        self.GCN_atom = GCNOne(self.args, edge_get_atom_features_dim() - 3)
        self.GCN_bond = GCNOne(self.args, edge_get_bond_features_dim() - 3)
        self.fpn = FPN(self.args).to(self.device)
        self.weights_atom = nn.Parameter(torch.rand(4, 1)).to(self.device)
        self.weights_bond = nn.Parameter(torch.rand(4, 1)).to(self.device)

        self.linear_dim = args.hidden_size
        self.dropout_fpn = args.dropout
        self.ffn_atom_bond = nn.Sequential(
            # nn.Dropout(self.dropout_fpn),
            # nn.Linear(in_features=linear_dim * 2, out_features=linear_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=self.linear_dim * 2, out_features=self.linear_dim * 1, bias=True)
        )

        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=self.linear_dim * 2, out_features=self.linear_dim * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=self.linear_dim * 2, out_features=args.task_num, bias=True)
        )
        self.is_classif = self.args.dataset_type == "classification"
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()
        # Move the model to the device
        self.ffn_atom_bond.to(self.device)
        self.ffn.to(self.device)


    def forward(self, smiles):
        atom_graph,bond_graph = create_atom_bond_graph(smiles, self.args, self.args.force_field)
        # atom_graph = create_atom_graph(smiles, self.args)
        # bond_graph = create_bond_graph(smiles, self.args)

        atoms_feature, atom_index = atom_graph.get_atom_feature() # Features, index records
        bonds_feature, bond_index = bond_graph.get_bond_feature()

        fpn_out = self.fpn(smiles)
        smiles_egnn_cls=[]

        atoms_feature = atoms_feature.cuda().to(self.device)
        bonds_feature = bonds_feature.cuda().to(self.device)
        fpn_out = fpn_out.to(self.device)

        for i, one in enumerate(smiles):  # Connection matrix
            mol = Chem.MolFromSmiles(one)
            atom2bond_list = []
            atom_edges = []
            for x, bond in enumerate(mol.GetBonds()): # Atom graph
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                atom2bond_list.append((x, (atom1, atom2))) # Record the edge index and the connected atoms
                atom_edges.extend([[atom1, atom2], [atom2, atom1]]) # Add molecular graph
            bond_edges = []
            for m in range(len(atom2bond_list)): # Bond graph
                for n in range(m + 1, len(atom2bond_list)): # Traverse pairwise, if there are common atoms, generate an edge
                    if set(atom2bond_list[m][1]) & set(atom2bond_list[n][1]):
                        # bond_edges.append((m, n)) #
                        bond_edges.extend([[m, n], [n, m]]) # Bidirectional
            atom_edges = torch.tensor(atom_edges, dtype=torch.long).t().contiguous()
            bond_edges = torch.tensor(bond_edges, dtype=torch.long).t().contiguous()  # New bond graph

            atom_edges = atom_edges.to(self.device)
            bond_edges = bond_edges.to(self.device)

            atom_start, atom_size = atom_index[i]
            bond_start, bond_size = bond_index[i]

            one_atom_feature = atoms_feature[atom_start:atom_start + atom_size]  # Features of the i-th molecule

            one_atom_feature_3d = one_atom_feature[:, -3:]
            one_atom_feature_scalar = one_atom_feature[:, :-3]

            # Feature convolution first
            one_atom_feature_scalar = self.GCN_atom(one_atom_feature_scalar, atom_edges)

            one_atom_feature_scalar, one_atom_feature_3d, global_atom_tokens = self.atom_3d_net(one_atom_feature_scalar, one_atom_feature_3d, atom_edges)

            # Apply 1D average pooling
            atom_concatenated_cls = torch.sum(global_atom_tokens[0] * self.weights_atom, dim=0, keepdim=True)

            # atom_concatenated_cls = torch.flatten(global_atom_tokens, start_dim=1)
            # bond_concatenated_cls = torch.flatten(global_bond_tokens, start_dim=1)
            if bond_size!=0:
                one_bond_feature = bonds_feature[bond_start:bond_start + bond_size]  # Features of the i-th molecule
                one_bond_feature_3d = one_bond_feature[:, -3:]
                one_bond_feature_scalar = one_bond_feature[:, :-3]  # Bond convolution and then merge into atom
                one_bond_feature_scalar = self.GCN_bond(one_bond_feature_scalar, bond_edges)
                transformer_bond_out = self.bond_3d_net(one_bond_feature_scalar) # Change to pure transformer encoder


            else:
                transformer_bond_out=torch.zeros(1,self.linear_dim).to(self.device)

            smiles_egnn_cls.append(torch.cat([atom_concatenated_cls,transformer_bond_out], axis=1))


        smiles_egnn_cls = torch.stack(smiles_egnn_cls, dim=1)[0]

        smiles_egnn_cls = self.ffn_atom_bond(smiles_egnn_cls)
        padded_batch_data = torch.cat([smiles_egnn_cls,fpn_out], axis=1)


        output = self.ffn(padded_batch_data)  # Define ffn according to different scales
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

        # Dimension transformation layer for residual connection
        self.residual_fc1 = nn.Linear(nfeat, self.atom_dim) if nfeat != self.atom_dim else nn.Identity()
        self.residual_fc2 = nn.Linear(self.atom_dim, self.atom_dim)
        # self.residual_fc3 = nn.Linear(self.atom_dim, self.atom_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the model to the device
        self.to(self.device)

    def forward(self, x, edge_index):
        # If the edge index is empty, directly use the fully connected layer
        if len(edge_index) == 0:
            hidden = self.fnn(x)
            return hidden

        # 1st Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)

        # Add residual connection
        residual = self.residual_fc1(x)
        hidden = torch.relu(hidden + residual)

        # 2nd Conv layer
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        # Add residual connection
        residual = self.residual_fc2(hidden)
        hidden = torch.relu(hidden + residual)

        # 3rd Conv layer
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        # Add residual connection
        # residual = self.residual_fc3(hidden)
        # hidden = hidden + residual

        # # 4th Conv layer
        # hidden = self.conv3(hidden, edge_index)
        # hidden = torch.tanh(hidden)

        return hidden

class EGNN_3d_net(nn.Module):
    def __init__(self, args):
        super(EGNN_3d_net, self).__init__()
        self.cuda = args.cuda
        self.egnnnet = EGNN_Network(
            dim=args.hidden_size,
            edge_dim=1,
            depth=3,
            num_nearest_neighbors=0,
            coor_weights_clamp_value=2.,
            global_linear_attn_every=1
            # Absolute clamped value for the coordinate weights, needed if you increase the num nearest neighbors
        )

        # self.egnn = EGNN(dim = args.hidden_size, edge_dim = 1)

        self.device = torch.device('cuda' if self.cuda else 'cpu')
        # Move the model to the device
        self.to(self.device)

    def atom_list2matrix(self, feats, edge_list):
        # Determine the number of nodes
        # print(feats.size())
        num_nodes = feats.size(0)
        if num_nodes == 1:
            return torch.ones(1, 1).to(feats.device)
        # Create an all-zero adjacency matrix
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        if edge_list.numel() != 0:
        # Fill the adjacency matrix
            for i in range(edge_list.shape[1]):  # Directly traverse each column
                src = edge_list[0, i]
                dst = edge_list[1, i]
                adj_matrix[src, dst] = 1  # Fill unidirectional relationship
                adj_matrix[dst, src] = 1  # Fill unidirectional relationship
        return adj_matrix

    def forward(self, feats, coors, edges):
        edges = self.atom_list2matrix(feats,edges)
        if self.cuda:
            edges = edges.cuda()
        # print(feats.unsqueeze(0).size())
        # print(coors.unsqueeze(0).size())
        # print(edges.unsqueeze(0).unsqueeze(-1).size())
        # print("=====")
        hidden = self.egnnnet(feats = feats.unsqueeze(0), coors = coors.unsqueeze(0), edges = edges.unsqueeze(0).unsqueeze(-1))
        return hidden


class FPN(nn.Module):  # Molecular fingerprint extraction, input SMILES, output 100-dimensional vector
    def __init__(self, args):
        super(FPN, self).__init__()
        self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.relu = nn.ReLU()
        self.hidden_dim = args.hidden_size
        self.args = args

        if hasattr(args, 'dataset_type'):
            self.fp_type = args.dataset_type  # 'classification'
        else:
            self.fp_type = 'mixed'

        if self.fp_type == 'classification':
            self.fp_dim = 3095
        if self.fp_type == 'regression':
            self.fp_dim = 1126

        if hasattr(args, 'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)
        self.cuda = args.cuda
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        # Move the model to the device
        self.to(self.device)

    def forward(self, smile):
        fp_list = []
        for i, one in enumerate(smile):
            fp = []
            mol = Chem.MolFromSmiles(one)
            if self.fp_type == 'classification':
                fp_PubChem = get_PubChemFingerprint([mol])
                fp_maccs = get_MACCSFingerprint([mol])
                fp_ECFPF = get_ECFPFingerprint([mol])
                fp.extend(fp_PubChem[0])
                fp.extend(fp_maccs[0])
                fp.extend(fp_ECFPF[0])

            if self.fp_type == 'regression':
                fp_PubChem = get_PubChemFingerprint([mol])
                fp_ERG = get_ERGFingerprint([mol])
                fp_EState = get_EStateFingerprint([mol])
                fp.extend(fp_PubChem[0])
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
        fpn_out = self.fc1(fp_list)
        fpn_out = self.relu(fpn_out)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out


if __name__ =="__main__":
    from model.tool import set_train_argument, set_predict_argument, set_hyper_argument, set_interfp_argument, \
        set_intergraph_argument
    args = set_train_argument()
    args.cuda = True
    args.num_folds = 10
    args.epochs = 20
    args.dataset_type='classification'
    args.save_path='model_save'
    args.model_type='moleculeformer'
    args.log_path='log'
    args.split_type = 'ratio_random'
    args.data_path = 'Data/MoleculeNet_dataset/bace.csv'
    args.dropout = 0
    model = moleculeformer(args)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    model(['C.C','CC1(C)CC[C@]2(NC(=O)C(C)(F)F)CC[C@]3(C)[C@H](C(=O)C=C4[C@@]3(C)CC[C@H]3C(C)(C)C(=O)C(C#N)=C[C@]43C)[C@@H]2C1'])