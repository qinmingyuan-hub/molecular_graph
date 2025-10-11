# -*- coding: UTF-8 -*-
'''
@Project ：Analysis_NER 
@File    ：EGNN_atom_bond.py
@Author  ：Mental-Flow
@Date    ：2024/11/14 11:36 
@introduction : 可变分子指纹
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

class TransformerEncoderWithCLS(nn.Module):
    def __init__(self, args, d_model=100, nhead=4, num_layers=2, dim_feedforward=200, dropout=0.1):
        super(TransformerEncoderWithCLS, self).__init__()
        # 定义可训练的 [CLS] 初始化 token
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.cls_token = nn.Parameter(torch.randn(1, d_model)).to(self.device)
        # 定义 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout).to(self.device)
        # 定义整个 Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

    def forward(self, src):
        # 在输入序列前拼接 [CLS] token

        src_with_cls = torch.cat([self.cls_token, src], dim=0)
        # 输入 src_with_cls 形状应为 (S, N, E)，S 是序列长度，N 是批量大小（这里是 1），E 是特征维度（这里是 100）
        output = self.transformer_encoder(src_with_cls)
        # 取出 [CLS] 对应的输出
        cls_output = output[0:1, :]
        return cls_output

class FPonly(nn.Module):
    def __init__(self, args):
        super(FPonly, self).__init__()
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
            nn.Linear(in_features=self.linear_dim * 1, out_features=self.linear_dim * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=self.linear_dim * 2, out_features=args.task_num, bias=True)
        )
        self.is_classif = self.args.dataset_type == "classification"
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()
        # 将模型移动到设备上
        self.ffn_atom_bond.to(self.device)
        self.ffn.to(self.device)


    def forward(self, smiles):
        fpn_out = self.fpn(smiles)

        padded_batch_data = torch.cat([fpn_out], axis=1)

        output = self.ffn(padded_batch_data)  # 根据不同的scale定义ffn
        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        return output

class GCNOne(nn.Module):  # 单个分子的GCN层
    def __init__(self, args, nfeat):
        super(GCNOne, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.atom_dim = args.hidden_size
        self.initial_conv = GCNConv(nfeat, self.atom_dim)
        self.fnn = nn.Linear(in_features=nfeat, out_features=self.atom_dim, bias=True)
        self.conv1 = GCNConv(self.atom_dim, self.atom_dim)
        self.conv2 = GCNConv(self.atom_dim, self.atom_dim)
        # self.conv3 = GCNConv(self.atom_dim, self.atom_dim)

        # 残差连接的维度变换层
        self.residual_fc1 = nn.Linear(nfeat, self.atom_dim) if nfeat != self.atom_dim else nn.Identity()
        self.residual_fc2 = nn.Linear(self.atom_dim, self.atom_dim)
        # self.residual_fc3 = nn.Linear(self.atom_dim, self.atom_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 将模型移动到设备上
        self.to(self.device)

    def forward(self, x, edge_index):
        # 如果边索引为空，直接使用全连接层
        if len(edge_index) == 0:
            hidden = self.fnn(x)
            return hidden

        # 1st Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)

        # 添加残差连接
        residual = self.residual_fc1(x)
        hidden = torch.relu(hidden + residual)

        # 2nd Conv layer
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        # 添加残差连接
        residual = self.residual_fc2(hidden)
        hidden = torch.relu(hidden + residual)

        # 3rd Conv layer
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        # 添加残差连接
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
            # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )

        # self.egnn = EGNN(dim = args.hidden_size, edge_dim = 1)

        self.device = torch.device('cuda' if self.cuda else 'cpu')
        # 将模型移动到设备上
        self.to(self.device)

    def atom_list2matrix(self, feats, edge_list):
        # 确定节点数量
        # print(feats.size())
        num_nodes = feats.size(0)
        if num_nodes == 1:
            return torch.ones(1, 1).to(feats.device)
        # 创建全零的邻接矩阵
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        if edge_list.numel() != 0:
        # 填充邻接矩阵
            for i in range(edge_list.shape[1]):  # 直接遍历每一列
                src = edge_list[0, i]
                dst = edge_list[1, i]
                adj_matrix[src, dst] = 1  # 填充单向关系
                adj_matrix[dst, src] = 1  # 填充单向关系
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


class FPN(nn.Module):  # 分子指纹提取 输入smile，分子指纹向量 输出100维度向量
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
            self.fp_dim = 1275

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
        # 将模型移动到设备上
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
    model = FPonly(args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params}")

    model(['C.C','CC1(C)CC[C@]2(NC(=O)C(C)(F)F)CC[C@]3(C)[C@H](C(=O)C=C4[C@@]3(C)CC[C@H]3C(C)(C)C(=O)C(C#N)=C[C@]43C)[C@@H]2C1'])