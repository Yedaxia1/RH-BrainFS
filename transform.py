from typing import Union, Tuple, Optional

import random
import torch
import scipy.sparse as ssp
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor
from torch_geometric.utils import sort_edge_index, get_laplacian, to_scipy_sparse_matrix
from scipy import linalg
from batch import Batch
from collections import defaultdict


class ToSparseTensor(BaseTransform):
    def __init__(self, attr: Optional[str] = 'edge_weight', fill_cache: bool = True):
        self.attr = attr
        self.fill_cache = fill_cache

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue
            if 'original_edge_index' in store:
                edge_key = 'original_edge_index'
                sparse_size = (store.original_x.size(0),store.original_x.size(0))
            else:
                edge_key = 'edge_index'
                sparse_size = store.size()[::-1]
                
            nnz = store[edge_key].size(1)

            keys, values = [], []
            for key, value in store.items():
                if isinstance(value, Tensor) and value.size(0) == nnz:
                    keys.append(key)
                    values.append(value)

            store[edge_key], values = sort_edge_index(store[edge_key],
                                                       values,
                                                       sort_by_row=False)

            for key, value in zip(keys, values):
                store[key] = value

            adj_t = SparseTensor(
                row=store[edge_key][1], col=store[edge_key][0],
                value=None if self.attr is None or self.attr not in store else
                store[self.attr], sparse_sizes=sparse_size,
                is_sorted=True)


            if self.fill_cache:  # Pre-process some important attributes.
                adj_t.storage.rowptr()
                adj_t.storage.csr2csc()

            store[edge_key] = adj_t.to_symmetric()
            if self.attr is not None and self.attr in store:
                del store[self.attr]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class RandomNodeSplit(BaseTransform):
    def __init__(
        self,
        split: str = "train_rest",
        num_splits: int = 1,
        num_train_per_class: int = 20,
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        key: Optional[str] = "y",
    ):
        assert split in ['train_rest', 'test_rest', 'random']
        self.split = split
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = key

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        if hasattr(store, "num_subgraphs"):
            num_nodes = store.num_subgraphs
        else:
            num_nodes = store.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_nodes * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round(num_nodes * self.num_test)
        else:
            num_test = self.num_test

        if self.split == 'train_rest':
            perm = torch.randperm(num_nodes)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True
        else:
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            for c in range(num_classes):
                idx = (y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                idx = idx[:self.num_train_per_class]
                train_mask[idx] = True

            remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            val_mask[remaining[:num_val]] = True

            if self.split == 'test_rest':
                test_mask[remaining[num_val:]] = True
            elif self.split == 'random':
                test_mask[remaining[num_val:num_val + num_test]] = True

        return train_mask, val_mask, test_mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(split={self.split})'

class HHopSubgraphs(BaseTransform):
    def __init__(self, h=1, max_nodes_per_hop=None, node_label='hop', use_rd=False, subgraph_pretransform=None):
        self.h = h
        self.max_nodes_per_hop = max_nodes_per_hop
        self.node_label = node_label
        self.use_rd = use_rd
        self.subgraph_pretransform = subgraph_pretransform

    def __call__(self, data):
        if type(self.h) == int:
            self.h = [self.h]
        assert(isinstance(data, Data))
        x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

        new_data_multi_hop = {}
        for h_ in self.h:
            subgraphs = []
            node_indice = []
            edge_indice = []
            for ind in range(num_nodes):
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, h_, edge_index, True, num_nodes, node_label=self.node_label,
                    max_nodes_per_hop=self.max_nodes_per_hop
                )
                node_indice.append(nodes_)
                x_ = None
                edge_attr_ = None
                pos_ = None

                if x is not None:
                    x_ = x[nodes_]

                if 'node_type' in data:
                    node_type_ = data.node_type[nodes_]

                if data.edge_attr is not None:
                    edge_attr_ = data.edge_attr[edge_mask_]
                if data.pos is not None:
                    pos_ = data.pos[nodes_]
                data_ = Data(x_, edge_index_, edge_attr_, None, pos_, z=z_)
                data_.num_nodes = nodes_.shape[0]

                if 'node_type' in data:
                    data_.node_type = node_type_

                if self.use_rd:
                    # See "Link prediction in complex networks: A survey".
                    adj = to_scipy_sparse_matrix(
                        edge_index_, num_nodes=nodes_.shape[0]
                    ).tocsr()
                    laplacian = ssp.csgraph.laplacian(adj).toarray()
                    try:
                        L_inv = linalg.pinv(laplacian)
                    except:
                        laplacian += 0.01 * np.eye(*laplacian.shape)
                    lxx = L_inv[0, 0]
                    lyy = L_inv[list(range(len(L_inv))), list(range(len(L_inv)))]
                    lxy = L_inv[0, :]
                    lyx = L_inv[:, 0]
                    rd_to_x = torch.FloatTensor(
                        (lxx + lyy - lxy - lyx)).unsqueeze(1)
                    data_.rd = rd_to_x

                if self.subgraph_pretransform is not None:  # for k-gnn
                    data_ = self.subgraph_pretransform(data_)
                    if 'assignment_index_2' in data_:
                        data_.batch_2 = torch.zeros(
                            data_.iso_type_2.shape[0], dtype=torch.long
                        )
                    if 'assignment_index_3' in data_:
                        data_.batch_3 = torch.zeros(
                            data_.iso_type_3.shape[0], dtype=torch.long
                        )

                subgraphs.append(data_)

            # new_data is treated as a big disconnected graph of the batch of subgraphs
            new_data = Batch.from_data_list(subgraphs)
            new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
            new_data.num_subgraphs = len(subgraphs)
            new_data.node_index = torch.cat(node_indice, dim=-1)

            new_data.original_edge_index = edge_index
            new_data.original_edge_attr = data.edge_attr
            new_data.original_pos = data.pos
            new_data.original_x = data.x

            # rename batch, because batch will be used to store node_to_graph assignment
            new_data.node_to_subgraph = new_data.batch
            del new_data.batch
            if 'batch_2' in new_data:
                new_data.assignment2_to_subgraph = new_data.batch_2
                del new_data.batch_2
            if 'batch_3' in new_data:
                new_data.assignment3_to_subgraph = new_data.batch_3
                del new_data.batch_3

            # create a subgraph_to_graph assignment vector (all zero)
            new_data.subgraph_to_graph = torch.zeros(
                len(subgraphs), dtype=torch.long)

            # copy remaining graph attributes
            for k, v in data:
                if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                            'z', 'rd', 'node_type']:
                    new_data[k] = v

            if len(self.h) == 1:
                return new_data
            else:
                new_data_multi_hop[h_] = new_data
        return new_data_multi_hop


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', node_label='hop',
                   max_nodes_per_hop=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    if node_label == 'hop':
        hops = [torch.LongTensor([0], device=row.device).flatten()]
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(h+2)
        if len(tmp) == 0:
            break
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(tmp):
                tmp = random.sample(tmp, max_nodes_per_hop)
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
        if node_label == 'hop':
            hops.append(torch.LongTensor(
                [h+1] * len(new_nodes), device=row.device))
    subset = torch.cat(subsets)
    inverse_map = torch.tensor(range(subset.shape[0]))
    if node_label == 'hop':
        hop = torch.cat(hops)
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    z = None
    if node_label == 'hop':
        hop = hop[hop != 0]
        hop = torch.cat([torch.LongTensor([0], device=row.device), hop])
        z = hop.unsqueeze(1)
    elif node_label.startswith('spd') or node_label == 'drnl':
        if node_label.startswith('spd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label == 'drnl':
            # see "Link Prediction Based on Graph Neural Networks", a special
            # case of spd2
            num_spd = 2
            z = torch.zeros([subset.size(0), 1],
                            dtype=torch.long, device=row.device)

        for i, node in enumerate(subset.tolist()):
            dists = label[node][:num_spd]  # keep top num_spd distances
            if node_label == 'spd':
                z[i][:min(num_spd, len(dists))] = torch.tensor(dists)
            elif node_label == 'drnl':
                dist1 = dists[0]
                dist2 = dists[1] if len(dists) == 2 else 0
                if dist2 == 0:
                    dist = dist1
                else:
                    dist = dist1 * (num_hops + 1) + dist2
                z[i][0] = dist

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask, z


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

class LapEncoding(BaseTransform):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr
    def __call__(self, data):
        num_node = data.num_nodes
        assert num_node == data.x.size(0)
        edge_attr = data.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(data.edge_index, edge_attr, normalization=self.normalization)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVec = np.real(EigVec[:,idx])[:, 1:self.pos_enc_dim+1]
        EigVec = torch.from_numpy(EigVec).float()
        EigVec = F.normalize(EigVec, p=2, dim=1, eps=1e-12, out=None)
        if num_node == 1:
            EigVec = torch.zeros((1, self.pos_enc_dim))
        elif num_node - 1 < self.pos_enc_dim:
            EigVec = F.pad(EigVec, (0, self.pos_enc_dim - num_node + 1), value=0.)
        assert EigVec.size(0) == num_node
        data.lpe = EigVec
        return data
    

class RandomAddEdge(BaseTransform):
    def __init__(self, p):
        self.p = p
    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.x.size(0)


        edge_set = set(map(tuple, edge_index.transpose(0, 1).tolist()))
        num_of_new_edge = int((edge_index.size(1) // 2) * self.p)
        to_add = list()
        new_edges = random.sample(range(1, num_nodes**2 + 1), num_of_new_edge + len(edge_set) + num_nodes)
        c = 0
        for i in new_edges:
            if c >= num_of_new_edge:
                break
            s = ((i - 1) // num_nodes) + 1
            t = i - (s - 1) * num_nodes
            s -= 1
            t -= 1
            if s != t and (s, t) not in edge_set:
                c += 1
                to_add.append([s, t])
                to_add.append([t, s])
                edge_set.add((s, t))
                edge_set.add((t, s))
        print(f"num of added edges: {len(to_add)}")
        new_edge_index = torch.cat([edge_index, torch.LongTensor(to_add).transpose(0, 1)], dim=1)
        data.edge_index = new_edge_index

        return data
