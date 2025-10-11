import numpy as np
import torch
import torch.nn as nn
from model.data import GetPubChemFPs, create_graph, get_atom_features_dim
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import GCNConv
# GAT + GCN + 3D + FingerPrint
atts_out = []


class GCN(nn.Module): #输入smile，输出GAT out
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.encoder = GCNEncoder(self.args)

    def forward(self, smile):
        mol = create_graph(smile, self.args)
        gcn_out = self.encoder.forward(mol, smile)

        return gcn_out

class GCNEncoder(nn.Module): #编码器
    def __init__(self, args):
        super(GCNEncoder, self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GCNOne(self.args)

    def forward(self, mols, smiles):
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()

        gat_outs = []
        for i, one in enumerate(smiles):
            adj = []
            mol = Chem.MolFromSmiles(one)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj / 1
            adj = torch.from_numpy(adj)
            if self.cuda:
                adj = adj.cuda()

            edge=[]
            for bond in mol.GetBonds():  # 扯淡么不是
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                edge.extend([[atom1, atom2], [atom2, atom1]])



            atom_start, atom_size = atom_index[i]
            one_feature = atom_feature[atom_start:atom_start + atom_size]
            edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
            if self.cuda:
                edge = edge.cuda()

            gat_atoms_out = self.encoder(one_feature, edge)  # 单个分子的GAT层
            gat_out = gat_atoms_out.sum(dim=0) / atom_size  # 平均池化。。
            gat_outs.append(gat_out)
        gat_outs = torch.stack(gat_outs, dim=0)
        return gat_outs


class GCNOne(nn.Module):  # 单个分子的GAT层
    def __init__(self, args):
        super(GCNOne, self).__init__()
        self.nfeat = get_atom_features_dim()
        self.atom_dim = args.hidden_size
        self.initial_conv = GCNConv(self.nfeat, self.atom_dim) #维度错误，后面写bach
        self.conv1 = GCNConv(self.atom_dim, self.atom_dim)
        self.conv2 = GCNConv(self.atom_dim, self.atom_dim)
        self.conv3 = GCNConv(self.atom_dim, self.atom_dim)

    def forward(self, x, edge_index):
        # 1st Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.tanh(hidden)

        # Other layers
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = torch.tanh(hidden)
        return hidden



class GCNModel(nn.Module): #组合的方法
    def __init__(self, is_classif, gat_scale, cuda, dropout_fpn):
        super(GCNModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()




    def create_gcn(self, args):
        self.encoder4 = GCN(args)


    def create_scale(self, args):
        linear_dim = int(args.hidden_size)

          # GAT维度,gat_scale控制参数比例
        self.fc_gat = nn.Linear(linear_dim, linear_dim)
        self.fc_fpn = nn.Linear(linear_dim, linear_dim)
        self.fc_gcn = nn.Linear(linear_dim, linear_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self, args):  # 根据不同的scale定义ffn（应该是最后一层），这里只修改了hidden_size
        linear_dim = args.hidden_size
        if self.gat_scale == 1:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )
        elif self.gat_scale == 0:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )

        else:
            self.ffn = nn.Sequential(
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.dropout_fpn),
                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
            )

    def forward(self, input):
        if self.gat_scale == 1:  # 根据不同的scale定义encoder，输出的out，到ffn里面
            output = self.encoder3(input)
        elif self.gat_scale == 0:
            output = self.encoder2(input)
        else:

            gcn_out =self.encoder4(input) #加一个GCN vout+GCNout=一个linear_dim长度
            gcn_out = self.fc_gcn(gcn_out)
            gcn_out = self.act_func(gcn_out)


            output = torch.cat([gcn_out], axis=1) #这里合在一起
        output = self.ffn(output)  # 根据不同的scale定义ffn

        if self.is_classif and not self.training: #如果是分类，就加一个sigmoid
            output = self.sigmoid(output)

        return output


def get_atts_out(): #不知道干啥的，留着 GAT相关
    return atts_out


def GCNGNN(args): #最后结合
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = GCNModel(is_classif, args.gat_scale, args.cuda, args.dropout)
    if args.gat_scale == 1:  # The ratio of gnn in model.默认0.5 ，不同的scale模型不一样，应该是消融实验，GAT和指纹
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:  # 全都要的一版

        model.create_gcn(args)

        model.create_scale(args)
        model.create_ffn(args)

    for param in model.parameters():  # 参数初始化
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model

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
    model = GCN(args)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params}")
