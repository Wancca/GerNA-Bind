import os
import torch
import numpy as np
from torch.utils import data
from torch_geometric.data import Batch
from utils.net_utils import pack1D, pack2D


class GerNA_FastDataset(data.Dataset):
    def __init__(self, pt_dir):
        self.pt_dir = pt_dir
        self.file_list = sorted(
            [f for f in os.listdir(pt_dir) if f.startswith('sample_') and f.endswith('.pt')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        self.num_samples = len(self.file_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        path = os.path.join(self.pt_dir, f"sample_{index}.pt")
        # 确保 weights_only=False 以加载自定义类
        return torch.load(path, weights_only=False)


def fast_collate_fn(batch):

    # 1. 提取各组件
    RNA_repre = [item['RNA_repre'] for item in batch]
    RNA_feats = [item['RNA_feats'] for item in batch]
    RNA_C4_coors = [item['RNA_C4_coors'] for item in batch]
    RNA_coors = [item['RNA_coors'] for item in batch]
    Mol_feats = [item['Mol_feats'] for item in batch]
    Mol_coors = [item['Mol_coors'] for item in batch]
    Mol_LAS = [item['Mol_LAS'] for item in batch]

    # 【核心修复点】：这些 Mask 长度也不一样，不能 stack，必须用 pack1D 对齐
    RNA_repre_mask_list = [item['RNA_repre_mask'] for item in batch]
    RNA_feats_mask_list = [item['RNA_feats_mask'] for item in batch]
    Mol_coors_mask_list = [item['Mol_coors_mask'] for item in batch]

   
    batch_label = torch.cat([item['label'] for item in batch])

    
    batch_RNA_repre = torch.FloatTensor(np.array(pack2D(RNA_repre)))
    batch_RNA_feats = torch.LongTensor(np.array(pack1D(RNA_feats)))
    batch_RNA_C4_coors = torch.FloatTensor(np.array(pack2D(RNA_C4_coors)))
    batch_RNA_coors = torch.FloatTensor(np.array(pack2D(RNA_coors)))
    batch_Mol_feats = torch.LongTensor(np.array(pack1D(Mol_feats)))
    batch_Mol_coors = torch.FloatTensor(np.array(pack2D(Mol_coors)))
    batch_Mol_LAS = torch.FloatTensor(np.array(pack2D(Mol_LAS)))

    # 对 Mask 进行对齐
    batch_seq_mask = torch.FloatTensor(np.array(pack1D(RNA_repre_mask_list)))
    batch_RNA_mask = torch.BoolTensor(np.array(pack1D(RNA_feats_mask_list)))
    batch_Mol_mask = torch.BoolTensor(np.array(pack1D(Mol_coors_mask_list)))

    # 3. 图数据打包 (PyG 对象)
    batch_Mol_Graph = Batch.from_data_list([item['Mol_graph'] for item in batch])
    batch_RNA_Graph = Batch.from_data_list([item['RNA_Graph'] for item in batch])

    # 返回顺序必须严格匹配 train_model.py 的解包顺序 (13个变量)
    return (batch_RNA_repre, batch_seq_mask, batch_Mol_Graph, batch_RNA_Graph,
            batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors,
            batch_RNA_mask, batch_Mol_feats, batch_Mol_coors,
            batch_Mol_mask, batch_Mol_LAS, batch_label)
