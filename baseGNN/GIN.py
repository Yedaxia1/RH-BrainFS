import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, in_size, nb_class, d_model, dropout=0.1, nb_layers=4):
        super(GIN, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout
        self.conv1 = GINConv(
            Sequential(
                Linear(self.features, self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
                ReLU(),
                BN(self.hidden_dim),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden_dim, self.hidden_dim),
                        ReLU(),
                        Linear(self.hidden_dim, self.hidden_dim),
                        ReLU(),
                        BN(self.hidden_dim),
                    ), train_eps=True))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def forward(self, data, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)

        x = self.fc_forward(x)

        return x

    def __repr__(self):
        return self.__class__.__name__
