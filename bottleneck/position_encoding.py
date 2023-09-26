import os
import pickle

import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from torch_geometric.data import Data

class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_sc_pe = []
        all_fc_pe = []
        dataset.sc_pe_list = []
        dataset.fc_pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                sc_g = Data(x=g.sc_x, edge_index=g.sc_edge_index, num_nodes=g.num_nodes, edge_attr=g.edge_attr)
                fc_g = Data(x=g.fc_x, edge_index=g.fc_edge_index, num_nodes=g.num_nodes, edge_attr=g.edge_attr)
                sc_pe, fc_pe = self.compute_pe(sc_g), self.compute_pe(fc_g)
                all_sc_pe.append(sc_pe)
                all_fc_pe.append(fc_pe)
            else:
                sc_pe = saved_pos_enc["all_sc_pe"][i]
                fc_pe = saved_pos_enc["all_fc_pe"][i]
            if self.zero_diag:
                sc_pe = sc_pe.clone()
                sc_pe.diagonal()[:] = 0

                fc_pe = fc_pe.clone()
                fc_pe.diagonal()[:] = 0

            dataset.sc_pe_list.append(sc_pe)
            dataset.fc_pe_list.append(fc_pe)

        self.save({"all_sc_pe":all_sc_pe, "all_fc_pe":all_fc_pe}, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass


class DiffusionEncoding(PositionEncoding):
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
                graph.edge_index, edge_attr, normalization=self.normalization,
                num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = expm(-self.beta * L)
        return torch.from_numpy(L.toarray())


class PStepRWEncoding(PositionEncoding):
    def __init__(self, savepath, p=1, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = sp.identity(L.shape[0], dtype=L.dtype) - self.beta * L
        tmp = L
        for _ in range(self.p - 1):
            tmp = tmp.dot(L)
        
        pe = torch.from_numpy(tmp.toarray())
        bias = torch.eye(graph.num_nodes) / 1e6
        pe = pe + bias
        return pe


class AdjEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        adj =  to_dense_adj(graph.edge_index)
        bias = torch.eye(graph.num_nodes) / 1e6
        adj = adj + bias
        return adj

class FullEncoding(PositionEncoding):
    def __init__(self, savepath, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)

    def compute_pe(self, graph):
        return torch.ones((graph.num_nodes, graph.num_nodes))

## Absolute position encoding
class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization)
        L = to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes=graph.num_nodes).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()

    def apply_to(self, dataset):
        dataset.sc_lap_pe_list = []
        dataset.fc_lap_pe_list = []
        for i, g in enumerate(dataset):
            sc_g = Data(x=g.sc_x, edge_index=g.sc_edge_index, num_nodes=g.num_nodes, edge_attr=g.edge_attr)
            fc_g = Data(x=g.fc_x, edge_index=g.fc_edge_index, num_nodes=g.num_nodes, edge_attr=g.edge_attr)
            sc_pe, fc_pe = self.compute_pe(sc_g), self.compute_pe(fc_g)
            # if fc_pe.shape[0] == 89:
            #     print("11") 
            dataset.sc_lap_pe_list.append(sc_pe)
            dataset.fc_lap_pe_list.append(fc_pe)

        return dataset


POSENCODINGS = {
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
    "adj": AdjEncoding,
}
