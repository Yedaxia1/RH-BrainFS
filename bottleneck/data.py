# from torch.utils.data import Datasets

from pickle import PicklingError
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data

class GraphDataset(object):
    def __init__(self, dataset, n_tags=None, degree=False):
        """a pytorch geometric dataset as input
        """
        self.dataset = dataset
        self.n_features_sc = dataset[0].sc_x.shape[-1]
        self.n_features_fc = dataset[0].fc_x.shape[-1]
        self.sc_pe_list = None
        self.fc_pe_list = None
        self.sc_lap_pe_list = None
        self.fc_lap_pe_list = None
        self.sc_degree_list = None
        self.fc_degree_list = None
        if degree:
            self.compute_degree()
        self.n_tags = n_tags
        # self.one_hot()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # if self.x_onehot is not None and len(self.x_onehot) == len(self.dataset):
        #     data.x_onehot = self.x_onehot[index]
        if self.sc_pe_list is not None and len(self.sc_pe_list) == len(self.dataset):
            data.sc_pe = self.sc_pe_list[index]
        if self.fc_pe_list is not None and len(self.fc_pe_list) == len(self.dataset):
            data.fc_pe = self.fc_pe_list[index]
        if self.sc_lap_pe_list is not None and len(self.sc_lap_pe_list) == len(self.dataset):
            data.sc_lap_pe = self.sc_lap_pe_list[index]
        if self.fc_lap_pe_list is not None and len(self.fc_lap_pe_list) == len(self.dataset):
            data.fc_lap_pe = self.fc_lap_pe_list[index]
        if self.sc_degree_list is not None and len(self.sc_degree_list) == len(self.dataset):
            data.sc_degree = self.sc_degree_list[index]
        if self.fc_degree_list is not None and len(self.fc_degree_list) == len(self.dataset):
            data.fc_degree = self.fc_degree_list[index]
        return data

    def compute_degree(self):
        self.sc_degree_list = []
        self.fc_degree_list = []
        for g in self.dataset:
            sc_deg = 1. / torch.sqrt(1. + utils.degree(g.sc_edge_index[0], g.num_nodes))
            self.sc_degree_list.append(sc_deg)

            fc_deg = 1. / torch.sqrt(1. + utils.degree(g.fc_edge_index[0], g.num_nodes))
            self.fc_degree_list.append(fc_deg)

    def input_size(self):
        if self.n_tags is None:
            return self.n_features
        return self.n_tags

    def one_hot(self):
        self.x_onehot = None
        if self.n_tags is not None and self.n_tags > 1:
            self.x_onehot = []
            for g in self.dataset:
                onehot = F.one_hot(g.x.view(-1).long(), self.n_tags)
                self.x_onehot.append(onehot)

    def collate_fn(self):
        def collate(batch):
            batch = list(batch)
            max_len_sc = max(len(g.sc_x) for g in batch)
            max_len_fc = max(len(g.fc_x) for g in batch)

            if self.n_tags is None:
                padded_x_sc = torch.zeros((len(batch), max_len_sc, self.n_features_sc))
                padded_x_fc = torch.zeros((len(batch), max_len_fc, self.n_features_fc))
            else:
                # discrete node attributes
                padded_x_sc = torch.zeros((len(batch), max_len_sc, self.n_tags))
                padded_x_fc = torch.zeros((len(batch), max_len_fc, self.n_tags))

            mask_sc = torch.zeros((len(batch), max_len_sc), dtype=bool)
            mask_fc = torch.zeros((len(batch), max_len_fc), dtype=bool)
            adjs_sc = torch.zeros((len(batch),max_len_sc, max_len_sc), dtype=torch.float32)
            adjs_fc = torch.zeros((len(batch),max_len_fc, max_len_fc), dtype=torch.float32)

            labels = []
            sc_edge_indice = []
            fc_edge_indice = []
            # TODO: check if position encoding matrix is sparse
            # if it's the case, use a huge sparse matrix
            # else use a dense tensor
            sc_pos_enc = None
            fc_pos_enc = None
            use_sc_pe = hasattr(batch[0], 'sc_pe') and batch[0].sc_pe is not None
            if use_sc_pe:
                if not batch[0].sc_pe.is_sparse:
                    sc_pos_enc = torch.zeros((len(batch), max_len_sc, max_len_sc))
                else:
                    print("Not implemented yet!")
            use_fc_pe = hasattr(batch[0], 'fc_pe') and batch[0].fc_pe is not None
            if use_fc_pe:
                if not batch[0].fc_pe.is_sparse:
                    fc_pos_enc = torch.zeros((len(batch), max_len_fc, max_len_fc))
                else:
                    print("Not implemented yet!")

            # process lap PE
            sc_lap_pos_enc = None
            fc_lap_pos_enc = None
            use_lap_pe = hasattr(batch[0], 'sc_lap_pe') and batch[0].sc_lap_pe is not None
            if use_lap_pe:
                sc_lap_pe_dim = batch[0].sc_lap_pe.shape[-1]
                sc_lap_pos_enc = torch.zeros((len(batch), max_len_sc, sc_lap_pe_dim))

                fc_lap_pe_dim = batch[0].fc_lap_pe.shape[-1]
                fc_lap_pos_enc = torch.zeros((len(batch), max_len_fc, fc_lap_pe_dim))

            sc_degree = None
            fc_degree = None
            use_degree = hasattr(batch[0], 'sc_degree') and batch[0].sc_degree is not None
            if use_degree:
                sc_degree = torch.zeros((len(batch), max_len_sc))

                fc_degree = torch.zeros((len(batch), max_len_fc))

            for i, g in enumerate(batch):

                labels.append(g.y)
                g_len_sc = len(g.sc_x)
                size_sc = torch.Size([g_len_sc, g_len_sc])
                g.sc_edge_attr = sc_edge_attr = torch.ones(g.sc_edge_index.size(1), dtype=torch.float)
                sc_edge_indice.append(g.sc_edge_index)
                sc_adj = torch.sparse_coo_tensor(g.sc_edge_index, sc_edge_attr, size_sc)
                sc_adj = sc_adj.to_dense()      
                adjs_sc[i, :g_len_sc, :g_len_sc] = sc_adj

                g_len_fc = len(g.fc_x)
                size_fc = torch.Size([g_len_fc, g_len_fc])
                g.fc_edge_attr = fc_edge_attr = torch.ones(g.fc_edge_index.size(1), dtype=torch.float)
                fc_edge_indice.append(g.fc_edge_index)
                fc_adj = torch.sparse_coo_tensor(g.fc_edge_index, fc_edge_attr, size_fc)
                fc_adj = fc_adj.to_dense()      
                adjs_fc[i, :g_len_fc, :g_len_fc] = fc_adj  

                if self.n_tags is None:
                    padded_x_sc[i, :g_len_sc, :] = g.sc_x
                    padded_x_fc[i, :g_len_fc, :] = g.fc_x
                else:
                    padded_x_sc[i, :g_len_sc, :] = g.x_onehot
                mask_sc[i, g_len_sc:] = True
                mask_fc[i, g_len_fc:] = True
                if use_sc_pe:
                    sc_pos_enc[i, :g.sc_pe.shape[-2], :g.sc_pe.shape[-1]] = g.sc_pe
                if use_sc_pe:
                    fc_pos_enc[i, :g.fc_pe.shape[-2], :g.fc_pe.shape[-1]] = g.fc_pe
                if use_lap_pe:
                    # print("g_len", g_len, "g.lap_pe.shape[-1]", g.lap_pe.shape[-1], "g.lap_pe:", g.lap_pe.shape)
                    sc_lap_pos_enc[i, :g_len_sc, :g.sc_lap_pe.shape[-1]] = g.sc_lap_pe
                    fc_lap_pos_enc[i, :g_len_fc, :g.fc_lap_pe.shape[-1]] = g.fc_lap_pe
                    
                if use_degree:
                    sc_degree[i, :g_len_sc] = g.sc_degree
                    fc_degree[i, :g_len_fc] = g.fc_degree
            
            data = Data(sc_x=padded_x_sc, fc_x=padded_x_fc, mask_sc=mask_sc, mask_fc=mask_fc, \
                sc_pos_enc=sc_pos_enc, fc_pos_enc=fc_pos_enc, sc_lap_pos_enc=sc_lap_pos_enc, fc_lap_pos_enc=fc_lap_pos_enc, \
                    adjs_sc=adjs_sc, adjs_fc=adjs_fc)

            return data, default_collate(labels)
        return collate
