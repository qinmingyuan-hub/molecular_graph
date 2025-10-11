# -*- coding: UTF-8 -*-
'''
@Project ：moleculeformer
@File    ：scikit_fingerprint.py
@Author  ：Mental-Flow
@Date    ：2025/6/5 15:56 
@introduction :
'''

from skfp.fingerprints import PubChemFingerprint,MACCSFingerprint,ERGFingerprint,ECFPFingerprint,RDKitFingerprint,AvalonFingerprint,EStateFingerprint,\
GhoseCrippenFingerprint

def get_PubChemFingerprint(smiles):  # 知名度高
    return PubChemFingerprint().transform(smiles)

def get_MACCSFingerprint(smiles):  # 166个预定义子结构键，强于规则化药效特征提取
    return MACCSFingerprint().transform(smiles)

def get_ERGFingerprint(smiles):  # 分子简化为拓扑学图，专为药物分子活性筛选设计，聚焦关键药效团与分子骨架的抽象表征，骨架敏感性任务
    return ERGFingerprint().transform(smiles)

def get_ECFPFingerprint(smiles):  # 基于原子局部环境扩展，擅长捕获药效团和子结构相似性。
    return ECFPFingerprint().transform(smiles)

def get_RDKitFingerprint(smiles):  # 200+个物化性质描述符（logP、极性表面积等），覆盖全局分子特性
    return RDKitFingerprint().transform(smiles)

def get_AvalonFingerprint(smiles):  # 兼顾子结构和路径指纹，对复杂环系表征优异
    return AvalonFingerprint().transform(smiles)

def get_EStateFingerprint(smiles):  # 量化原子电性状态，擅长捕获电子效应（如氢键位点）
    return EStateFingerprint().transform(smiles)

# def get_ElectroShapeFingerprint(smiles):  # 3D指纹，需构象生成增加变量复杂度
#     fp = ElectroShapeFingerprint()
#     mol_from_smiles = MolFromSmilesTransformer()
#     mols = mol_from_smiles.transform(smiles)
#     conf_gen = ConformerGenerator()
#     mols = conf_gen.transform(mols)
#     return fp.transform(mols)

def get_GhoseCrippenFingerprint(smiles): #基于原子类型logP贡献，专长亲脂性/溶解性预测。直接关联脂水分配系数，适用于ADMET消融实验
    return GhoseCrippenFingerprint().transform(smiles)

if __name__ =="__main__":
    smiles = ["C.C"]
    # PubChem = PubChemFingerprint()
    # MACCS = MACCSFingerprint()
    # ERG = ERGFingerprint()
    # ECFP = ECFPFingerprint()
    # RDKit = RDKitFingerprint()
    # print(PubChem.transform(smiles).shape)  # 881
    # print(MACCS.transform(smiles).shape)  # 166
    # print(ERG.transform(smiles).shape)  # 315
    # print(ECFP.transform(smiles).shape)  # 2048
    # print(RDKit.transform(smiles).shape)  # 2048
    print(get_EStateFingerprint(smiles).shape)  # 79

    # print(get_ElectroShapeFingerprint(smiles))
