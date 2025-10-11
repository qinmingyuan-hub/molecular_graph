import numpy as np
import torch
import torch.nn as nn
from model.data import GetPubChemFPs, create_graph, get_atom_features_dim
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import GCNConv

class FPN(nn.Module):
    def __init__(self, args):
        super(FPN, self).__init__()
        self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args
        if hasattr(args, 'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'

        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024

        if hasattr(args, 'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fp_list = []
        for i, one in enumerate(smile):
            fp = []
            mol = Chem.MolFromSmiles(one)

            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
                fp_pubcfp = GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)

        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:, self.fp_changebit - 1] = np.ones(fp_list[:, self.fp_changebit - 1].shape)
            fp_list.tolist()

        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.encoder = GCNEncoder(self.args)

    def forward(self, smile):
        mol = create_graph(smile, self.args)
        gcn_out = self.encoder.forward(mol, smile)

        return gcn_out


class GCNEncoder(nn.Module):
    def __init__(self, args):
        super(GCNEncoder, self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GCNOne(self.args)

    def collate_fn(self,gat_outs):
        tensors = [item for item in gat_outs]
        max_seq_length = max([tensor.size(0) for tensor in tensors]) #最大长度？

        padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_seq_length - tensor.size(0))) for tensor in
                          tensors]

        attention_masks = []
        for tensor in tensors:
            seq_length , _ = tensor.size()
            mask = torch.zeros(max_seq_length+1, dtype=torch.bool)
            mask[seq_length+1:] = 1
            attention_masks.append(mask)

        return torch.stack(padded_tensors), torch.stack(attention_masks)

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
            for bond in mol.GetBonds():  # 边
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                edge.extend([[atom1, atom2], [atom2, atom1]])



            atom_start, atom_size = atom_index[i]
            one_feature = atom_feature[atom_start:atom_start + atom_size]
            edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
            if self.cuda:
                edge = edge.cuda()

            gat_atoms_out = self.encoder(one_feature, edge)

            # gat_out = gat_atoms_out.sum(dim=0) / atom_size
            gat_outs.append(gat_atoms_out)

        padded_batch_data, attention_masks = self.collate_fn(gat_outs)

        return padded_batch_data, attention_masks


class GCNOne(nn.Module):
    def __init__(self, args):
        super(GCNOne, self).__init__()
        self.nfeat = get_atom_features_dim()
        self.atom_dim = args.hidden_size
        self.initial_conv = GCNConv(self.nfeat, self.atom_dim)
        self.conv1 = GCNConv(self.atom_dim, self.atom_dim)
        self.conv2 = GCNConv(self.atom_dim, self.atom_dim)
        self.conv3 = GCNConv(self.atom_dim, self.atom_dim)

    def forward(self, x, edge_index): #？
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

class TransformerEncoder(nn.Module):
    def __init__(self,args):
        super(TransformerEncoder, self).__init__()

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=args.nheads,norm_first=True)

    def forward(self, input_data,mask):

        input_data = input_data.permute(1, 0, 2)

        encoder_output = self.transformer_encoder(input_data,src_key_padding_mask=mask.cuda()) #这里加了mask

        output = encoder_output[0]
        return output

class GCNTransformerModel(nn.Module):
    def __init__(self, is_classif, gat_scale, cuda, dropout_fpn):
        super(GCNTransformerModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 100))
    def create_transformer(self, args):
        self.encoder1 = TransformerEncoder(args)

    def create_fpn(self, args):
        self.encoder2 = FPN(args)

    def create_gcn(self, args):
        self.encoder4 = GCN(args)


    def create_scale(self, args):
        linear_dim = int(args.hidden_size)

        # GAT维度,gat_scale控制参数比例
        self.fc_fpn = nn.Linear(linear_dim, linear_dim)
        self.fc_gcn = nn.Linear(linear_dim, linear_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self, args):  #
        linear_dim = args.hidden_size
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
        )

    def forward(self, input): # 定义transformer

        padded_batch_data, attention_masks = self.encoder4(input) # 50,54,100
        cls_tokens = self.cls_token.expand(padded_batch_data.size()[0], -1, -1)
        # print(cls_tokens)
        # random_tensor = torch.randn(padded_batch_data.size()[0],1,300).cuda()
        padded_batch_data = torch.cat([cls_tokens, padded_batch_data], dim=1)

        transformer_out = self.encoder1(padded_batch_data,attention_masks)
        # print(transformer_out)
        output = torch.cat([transformer_out], axis=1)

        output = self.ffn(output)

        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output

def GCNTransformer(args): #最后结合
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = GCNTransformerModel(is_classif, args.gat_scale, args.cuda, args.dropout)

    model.create_transformer(args)
    model.create_fpn(args)
    model.create_gcn(args)
    model.create_scale(args)
    model.create_ffn(args)

    for param in model.parameters():  # 参数初始化
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model
