import os
from batch import Batch
import numpy as np
import torch
from scipy import io
from construct import *
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from scipy import sparse as sp
import dgl
import hashlib
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def decide_dataset(root):  # 返回标签
    if 'ABIDE' in root:
        class_dict = {
            "HC": 0,
            "ASD": 1,
        }
    elif 'Multi' in root or "xinxiang" in root or "zhongda" in root:
        class_dict = {
            "HC": 0,
            "MDD": 1,
        }
    elif "HCP" in root:
        class_dict = {
            "female": 0,
            "male": 1,
        }
    return class_dict


def read_dataset_fc_regionSeries(root, label_files, files):  # 返回时间序列
    FC_dir = "RegionSeries.mat"
    if 'ABIDE' in root:
        subj_fc_dir = os.path.join(root, label_files, files)
        subj_mat_fc = np.loadtxt(subj_fc_dir)[:176, :90]
    # elif 'multi' in root:
    #     subj_fc_dir = os.path.join(root, label_files, files)
    #     subj_mat_fc = io.loadmat(subj_fc_dir)['ROISignals_AAL'][:170, :]
    else:  # zhonda、xinxiang、ASD的读法都一样
        subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
        subj_mat_fc = io.loadmat(subj_fc_dir)['RegionSeries']
    return subj_mat_fc

def read_dataset_fc_edgeWeight(root, label_files, files):  # 返回时间序列
    FC_dir = "EdgeWeight.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['EdgeWeight']
    return subj_mat_fc

def read_dataset_fc_region_features(root, label_files, files):  # 返回时间序列
    FC_dir = "region_features_norm.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['region_features']
    return subj_mat_fc

def read_dataset_sc_connectivity(root, label_files, files):  # 返回sc connectivity
    SC_connectivity_dir = "DTI_connectivity_count.mat"
    subj_sc_connectivity_dir = os.path.join(root, label_files, files, SC_connectivity_dir)
    subj_sc_connectivity = io.loadmat(subj_sc_connectivity_dir)['connectivity']
    return subj_sc_connectivity


def read_dataset_sc_features(root, label_files, files):  # 返回sc特征
    SC_features_dir = "DTI_connectivity_count.mat"
    subj_sc_feature_dir = os.path.join(root, label_files, files, SC_features_dir)
    subj_sc_feature = io.loadmat(subj_sc_feature_dir)['connectivity']
    return subj_sc_feature



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.num_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return lap_pos_enc


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    wl_pos_enc = torch.LongTensor(list(node_color_dict.values()))
    return wl_pos_enc

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, args=None):
        self.args = args
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        
        data_sc_list = []
        data_fc_list = []
        
        self.sc = []
        self.fc = []
        self.scfc = []
        self.class_dict = decide_dataset(self.args.path)
        self.y = []

        root = self.args.path
        label_list = os.listdir(root)
        label_list.sort()
        maxx = 0

        for label_files in label_list:
            list = os.listdir(os.path.join(root, label_files))
            list.sort()
            label = torch.LongTensor([self.class_dict[label_files]])
            for files in list:
                ############
                sc_feature = read_dataset_sc_features(root, label_files, files)
                sc_connectivity = read_dataset_sc_connectivity(root, label_files, files)
                
                subj_sc_feature = sc_feature  # 构造特征
                subj_sc_adj = get_sc_adj(sc_connectivity, self.args)  # 构造邻接矩阵

                ############ 230*90
                fc_feature = read_dataset_fc_regionSeries(root, label_files, files)                 # 230*90
                fc_connectivity = read_dataset_fc_regionSeries(root, label_files, files)            # 230*90
                # fc_region_features = read_dataset_fc_region_features(root, label_files, files)
                
                subj_fc_feature = np.corrcoef(np.transpose(fc_feature))
                subj_fc_adj = get_Pearson_fc(fc_connectivity, self.args)  
                
                data_sc_list.append(get_sc_dataloader(subj_sc_adj, subj_sc_feature))            
                data_fc_list.append(get_fc_dataloader(subj_fc_adj, subj_fc_feature))
                self.y.append(label)

        if self.pre_filter is not None:
            data_sc_list = [data for data in data_sc_list if self.pre_filter(data)]
            data_fc_list = [data for data in data_fc_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            new_data_sc_list = []
            for data in tqdm(data_sc_list):
                new_data_sc_list.append(self.pre_transform(data))
            data_sc_list = new_data_sc_list
                
            new_data_fc_list = []
            for data in tqdm(data_fc_list):
                new_data_fc_list.append(self.pre_transform(data))
            data_fc_list = new_data_fc_list
            
        for _, (sc, fc) in enumerate(zip(data_sc_list, data_fc_list)):
            data = Batch()
            for key in sc.keys:
                data.__setattr__('sc_{}'.format(key), sc.__getattr__(key))
            
            for key in fc.keys:
                data.__setattr__('fc_{}'.format(key), fc.__getattr__(key))
                
            data.y = self.y[_]
            
            data_list.append(data)
            
                
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
