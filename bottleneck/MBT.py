
from bottleneck.layers.encoders import DiffTransformerEncoder, FusionTransformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, TopKPooling, global_add_pool,
                                JumpingKnowledge)
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import dgl
from einops import rearrange, repeat
from .layers.layers import *

'''
    AAA-BBB pattern 
'''
class Attention_Bottlenecks(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_heads = args.num_heads
        self.out_dim = args.hidden_dim
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.fusion_layers = args.fusion_layers
        self.num_bottlenecks = args.num_bottlenecks

        self.readout = args.readout
        self.pos_enc = args.pos_enc
        self.pos_enc_dim = args.pos_enc_dim
        self.lappe = args.lappe
        self.lap_dim = args.lap_dim
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.device = args.device
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward is not None else 2048

        self.cls_token_sc = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.cls_token_fc = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        self.Bottlnecks = nn.Parameter(torch.randn(1, self.num_bottlenecks, self.hidden_dim))

        self.embedding_sc = nn.Linear(self.sc_features, self.hidden_dim)
        self.embedding_fc = nn.Linear(self.fc_features, self.hidden_dim)

        self.embedding_sc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        self.embedding_fc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)

        # sc_encoder_layer = nn.TransformerEncoderLayer(
        #     self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True)
        # self.sc_encoder = nn.TransformerEncoder(sc_encoder_layer, self.num_layers - self.fusion_layers)

        # fc_encoder_layer = nn.TransformerEncoderLayer(
        #     self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True)
        # self.fc_encoder = nn.TransformerEncoder(fc_encoder_layer, self.num_layers - self.fusion_layers)

        sc_encoder_layer = DiffTransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_norm=self.batch_norm)
        self.sc_encoder = DiffTransformerEncoder(sc_encoder_layer, self.num_layers - self.fusion_layers)

        fc_encoder_layer = DiffTransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_norm=self.batch_norm)
        self.fc_encoder = DiffTransformerEncoder(fc_encoder_layer, self.num_layers - self.fusion_layers)

        fusion_encoder_layer = FusionTransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True)
        self.fusion_encoder = FusionTransformerEncoder(fusion_encoder_layer, self.fusion_layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )


    def forward(self, data):# h_lap_pos_enc=None):
        sc_x, fc_x, sc_pos_enc, fc_pos_enc, sc_lap_pos_enc, fc_lap_pos_enc = data.sc_x, data.fc_x, data.sc_pos_enc, data.fc_pos_enc, data.sc_lap_pos_enc, data.fc_lap_pos_enc
        
        b_sc, n_sc, _ = sc_x.shape
        b_fc, n_fc, _ = fc_x.shape

        ## 线性变化 --> hidden_dim
        sc_x = self.embedding_sc(sc_x)
        fc_x = self.embedding_fc(fc_x)

        if self.lappe and sc_lap_pos_enc is not None and fc_lap_pos_enc is not None:
            sc_pos_emb =  self.embedding_sc_lap_pos_enc(sc_lap_pos_enc)
            fc_pos_emb =  self.embedding_fc_lap_pos_enc(fc_lap_pos_enc)
            sc_x += sc_pos_emb
            fc_x += fc_pos_emb

        ## 增加<CLS>
        cls_token_sc = repeat(self.cls_token_sc, '1 1 d -> b 1 d', b = b_sc)
        sc_x = torch.cat((cls_token_sc, sc_x), dim=1)
        cls_token_fc = repeat(self.cls_token_fc, '1 1 d -> b 1 d', b = b_fc)
        fc_x = torch.cat((cls_token_fc, fc_x), dim=1)

        ## 前num_layers-mid_layer层做单模态Transformer
        sc_x = sc_x.permute(1, 0, 2)
        fc_x = fc_x.permute(1, 0, 2)
        sc_x = self.sc_encoder(sc_x, sc_pos_enc, degree=None)
        fc_x = self.fc_encoder(fc_x, fc_pos_enc, degree=None)
        sc_x = sc_x.permute(1, 0, 2)
        fc_x = fc_x.permute(1, 0, 2)

        single_modal_cls_sc = sc_x[:,0:1,:]
        single_modal_cls_fc = fc_x[:,0:1,:]

        ## 后fusion_layers层做信息共享
        b_bottlenecks = repeat(self.Bottlnecks, '1 n d -> b n d', b = b_sc)
        sc_x_res, fc_x_res, bottlenecks_res = self.fusion_encoder(sc_x, fc_x, b_bottlenecks)

        ## 取两个<CLS>进入同一个classifier，平均一下softmax的logits
        cls_sc = sc_x_res[:,0:1,:]
        cls_fc = fc_x_res[:,0:1,:]

        cls_sc_logits = self.classifier(cls_sc)
        cls_fc_logits = self.classifier(cls_fc)

        final_logit = (cls_sc_logits + cls_fc_logits)/2
        final_logit = torch.squeeze(final_logit, dim=1)

        return final_logit, single_modal_cls_sc, single_modal_cls_fc, cls_sc, cls_fc


'''
    AB-AB-AB pattern
'''
class Alternately_Attention_Bottlenecks(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_cuda = args.use_cuda
        
        self.num_heads = args.num_heads
        self.out_dim = args.hidden_dim
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_bottlenecks = args.num_bottlenecks

        self.readout = args.readout
        self.pos_enc = args.pos_enc
        self.pos_enc_dim = args.pos_enc_dim
        self.lappe = args.lappe
        self.lap_dim = args.lap_dim
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward is not None else 2048

        self.Bottlnecks = nn.Parameter(torch.randn(1, self.num_bottlenecks, self.hidden_dim))
        
        self.node_embedding_sc = nn.Linear(self.sc_features, self.hidden_dim)
        self.node_embedding_fc = nn.Linear(self.fc_features, self.hidden_dim)

        # self.embedding_sc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        # self.embedding_fc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)

        self.sc_subGraph_convs = nn.ModuleList()
        self.fc_subGraph_convs = nn.ModuleList()
        self.fusion_transformer_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
        self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
        self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
        
        self.multi_scale_fuse = nn.MultiheadAttention(self.hidden_dim, self.num_heads, self.dropout, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data):# h_lap_pos_enc=None):
        batch_size = data.batch.max().item() + 1
        
        sc_x, sc_sub_x, sc_edge_index, sc_sub_x_index, sc_sub_edge_idx, sc_node_to_subgraph, sc_subgraph_to_graph = self.process(data=data, modality='sc')
        fc_x, fc_sub_x, fc_edge_index, fc_sub_x_index, fc_sub_edge_idx, fc_node_to_subgraph, fc_subgraph_to_graph = self.process(data=data, modality='fc')
        
        b_bottlenecks = repeat(self.Bottlnecks, '1 n d -> b n d', b = batch_size)
        batch = torch.tensor([[i] * self.num_bottlenecks for i in range(batch_size)]).view(-1)
        if self.use_cuda:
            batch = batch.cuda()
        
        multi_scale_bottlenecks = []

        for i in range(self.num_layers-1):
            sc_x = self.sc_subGraph_convs[i](sc_sub_x, sc_sub_edge_idx, sc_node_to_subgraph)
            fc_x = self.fc_subGraph_convs[i](fc_sub_x, fc_sub_edge_idx, fc_node_to_subgraph)
            
            sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            
            sc_x, fc_x, b_bottlenecks = self.fusion_transformer_convs[i](sc_x, fc_x, b_bottlenecks)
            
            sc_x = sc_x.contiguous().view(-1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(-1, self.hidden_dim)
            b_bottlenecks = b_bottlenecks.contiguous().view(-1, self.hidden_dim)
            
            sc_x, fc_x, b_bottlenecks = self.bns[i](sc_x), self.bns[i](fc_x), self.bns[i](b_bottlenecks)
            sc_x, fc_x, b_bottlenecks = F.relu(sc_x), F.relu(fc_x), F.relu(b_bottlenecks)
            sc_x, fc_x, b_bottlenecks = F.dropout(sc_x, p=self.dropout, training=self.training), F.dropout(fc_x, p=self.dropout, training=self.training), F.dropout(b_bottlenecks, p=self.dropout, training=self.training)

            pooled_bottlenecks = global_mean_pool(b_bottlenecks, batch)
            multi_scale_bottlenecks.append(pooled_bottlenecks)
            
            b_bottlenecks = b_bottlenecks.contiguous().view(batch_size, -1, self.hidden_dim)

            sc_sub_x = sc_x[sc_sub_x_index] #get_subx(x, sub_x_idx, subgraph_to_graph, data.batch)
            fc_sub_x = fc_x[fc_sub_x_index]
        
        sc_x = self.sc_subGraph_convs[-1](sc_sub_x, sc_sub_edge_idx, sc_node_to_subgraph)
        fc_x = self.fc_subGraph_convs[-1](fc_sub_x, fc_sub_edge_idx, fc_node_to_subgraph)
        
        sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        sc_x, fc_x, b_bottlenecks = self.fusion_transformer_convs[-1](sc_x, fc_x, b_bottlenecks)
        
        sc_x, fc_x, b_bottlenecks = sc_x.contiguous(), fc_x.contiguous(), b_bottlenecks.contiguous()
        
        b_bottlenecks = b_bottlenecks.contiguous().view(-1, self.hidden_dim)
        pooled_bottlenecks = global_mean_pool(b_bottlenecks, batch)
        
        multi_scale_bottlenecks.append(pooled_bottlenecks)  # num_layers * bz * hidden_dim
        
        # x = self.bns[-1](x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        
        ## 最后一个b_bottlenecks做classifier
        logit = self.classifier(multi_scale_bottlenecks[-1].unsqueeze(1))
        logit = torch.squeeze(logit, dim=1)
        
        # ## 多尺度b_bottlenecks做classifier
        # # 1. list --> tensor --> view
        # multi_scale_bottlenecks = torch.stack(multi_scale_bottlenecks, dim=0).permute(1,0,2)
        # # 2. 自注意力计算
        # multi_scale_bottlenecks, attn_output_weights  = self.multi_scale_fuse(multi_scale_bottlenecks, multi_scale_bottlenecks, multi_scale_bottlenecks)
        # # 3. 平均
        # multi_scale_bottlenecks = torch.mean(multi_scale_bottlenecks, dim=1).unsqueeze(1)
        # # 4. 分类
        # logit = self.classifier(multi_scale_bottlenecks).squeeze(1)
        
        return logit, sc_x, fc_x, multi_scale_bottlenecks[-1].unsqueeze(1)    # (logit: bz * label_num;) (sc_x, fc_x: bz * node_num * hidden_dim)
    
    def process(self, data, modality='sc'):
        edge_index, x = data.__getattr__('{}_original_edge_index'.format(modality)),  data.__getattr__('{}_original_x'.format(modality))
        sub_x_index, sub_edge_idx = data.__getattr__('{}_node_index'.format(modality)), data.__getattr__('{}_edge_index'.format(modality)) 
        node_to_subgraph, subgraph_to_graph = data.__getattr__('{}_node_to_subgraph'.format(modality)), data.__getattr__('{}_subgraph_to_graph'.format(modality)) 

        x = eval('self.node_embedding_{}'.format(modality))(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        sub_x = x[sub_x_index] 
        
        sub_x = F.dropout(sub_x, p= self.dropout, training=self.training)
            
        return x, sub_x, edge_index, sub_x_index, sub_edge_idx, node_to_subgraph, subgraph_to_graph


class GNN_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_cuda = args.use_cuda
        
        self.num_heads = args.num_heads
        self.out_dim = args.hidden_dim
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_bottlenecks = args.num_bottlenecks

        self.readout = args.readout
        self.pos_enc = args.pos_enc
        self.pos_enc_dim = args.pos_enc_dim
        self.lappe = args.lappe
        self.lap_dim = args.lap_dim
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward is not None else 2048

        self.Bottlnecks = nn.Parameter(torch.randn(1, self.num_bottlenecks, self.hidden_dim))
        
        self.node_embedding_sc = nn.Linear(self.sc_features, self.hidden_dim)
        self.node_embedding_fc = nn.Linear(self.fc_features, self.hidden_dim)

        # self.embedding_sc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        # self.embedding_fc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )

        self.sc_subGraph_convs = nn.ModuleList()
        self.fc_subGraph_convs = nn.ModuleList()
        self.fusion_transformer_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            # self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            # self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            # self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            self.sc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
            self.fc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
            self.fusion_transformer_convs.append(nn.TransformerEncoderLayer(
                self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.sc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
        self.fc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
        self.fusion_transformer_convs.append(nn.TransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
    
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data):# h_lap_pos_enc=None):
        batch_size = data.batch.max().item() + 1
        
        sc_x, sc_sub_x, sc_edge_index, sc_sub_x_index, sc_sub_edge_idx, sc_node_to_subgraph, sc_subgraph_to_graph = self.process(data=data, modality='sc')
        fc_x, fc_sub_x, fc_edge_index, fc_sub_x_index, fc_sub_edge_idx, fc_node_to_subgraph, fc_subgraph_to_graph = self.process(data=data, modality='fc')
        
        batch = torch.tensor([[i] * 90 for i in range(batch_size)]).view(-1)
        if self.use_cuda:
            batch = batch.cuda()
        
        multi_scale_bottlenecks = []

        for i in range(self.num_layers-1):
            sc_x = self.sc_subGraph_convs[i](sc_x, sc_edge_index)
            fc_x = self.fc_subGraph_convs[i](fc_x, fc_edge_index)
            
            sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            
            fusion_x = torch.cat((sc_x, fc_x), dim=1)
            
            fusion_x = self.fusion_transformer_convs[i](fusion_x)
            
            sc_x = fusion_x[:,:90,:]
            sc_x = fusion_x[:,90:,:]
            
            sc_x = sc_x.contiguous().view(-1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(-1, self.hidden_dim)

            
            sc_x, fc_x = self.bns[i](sc_x), self.bns[i](fc_x)
            sc_x, fc_x = F.relu(sc_x), F.relu(fc_x)
            sc_x, fc_x = F.dropout(sc_x, p=self.dropout, training=self.training), F.dropout(fc_x, p=self.dropout, training=self.training)
            
            # pooled_bottlenecks = global_mean_pool(b_bottlenecks, batch)
            # multi_scale_bottlenecks.append(pooled_bottlenecks)
            
            # b_bottlenecks = b_bottlenecks.contiguous().view(batch_size, -1, self.hidden_dim)

            # sc_sub_x = sc_x[sc_sub_x_index] #get_subx(x, sub_x_idx, subgraph_to_graph, data.batch)
            # fc_sub_x = fc_x[fc_sub_x_index]
        
        sc_x = self.sc_subGraph_convs[-1](sc_x, sc_edge_index)
        fc_x = self.fc_subGraph_convs[-1](fc_x, fc_edge_index)
        
        sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        
        fusion_x = torch.cat((sc_x, fc_x), dim=1)
        
        fusion_x = self.fusion_transformer_convs[i](fusion_x)
        
        sc_x = fusion_x[:,:90,:]
        sc_x = fusion_x[:,90:,:]
        
        sc_x = sc_x.contiguous().view(-1, self.hidden_dim)
        fc_x = fc_x.contiguous().view(-1, self.hidden_dim)
        
        pooled_sc = global_mean_pool(sc_x, batch)
        pooled_fc = global_mean_pool(fc_x, batch)
        
        pooled_sc = pooled_sc.unsqueeze(1)
        pooled_fc = pooled_fc.unsqueeze(1)
        
        # x = self.bns[-1](x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        
        ## 最后一个b_bottlenecks做classifier
        logit_sc = self.classifier(pooled_sc)
        logit_fc = self.classifier(pooled_fc)
        logit = (logit_sc + logit_fc)/2
        logit = torch.squeeze(logit, dim=1)
        
        # ## 多尺度b_bottlenecks做classifier
        # # 1. list --> tensor --> view
        # multi_scale_bottlenecks = torch.stack(multi_scale_bottlenecks, dim=0).permute(1,0,2)
        # # 2. 自注意力计算
        # multi_scale_bottlenecks, attn_output_weights  = self.multi_scale_fuse(multi_scale_bottlenecks, multi_scale_bottlenecks, multi_scale_bottlenecks)
        # # 3. 平均
        # multi_scale_bottlenecks = torch.mean(multi_scale_bottlenecks, dim=1).unsqueeze(1)
        # # 4. 分类
        # logit = self.classifier(multi_scale_bottlenecks).squeeze(1)
        
        return logit, sc_x, fc_x    # (logit: bz * label_num;) (sc_x, fc_x: bz * node_num * hidden_dim)
    
    def process(self, data, modality='sc'):
        edge_index, x = data.__getattr__('{}_original_edge_index'.format(modality)),  data.__getattr__('{}_original_x'.format(modality))
        sub_x_index, sub_edge_idx = data.__getattr__('{}_node_index'.format(modality)), data.__getattr__('{}_edge_index'.format(modality)) 
        node_to_subgraph, subgraph_to_graph = data.__getattr__('{}_node_to_subgraph'.format(modality)), data.__getattr__('{}_subgraph_to_graph'.format(modality)) 

        x = eval('self.node_embedding_{}'.format(modality))(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        sub_x = x[sub_x_index] 
        
        sub_x = F.dropout(sub_x, p= self.dropout, training=self.training)
            
        return x, sub_x, edge_index, sub_x_index, sub_edge_idx, node_to_subgraph, subgraph_to_graph

class SubGraphGNN_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_cuda = args.use_cuda
        
        self.num_heads = args.num_heads
        self.out_dim = args.hidden_dim
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_bottlenecks = args.num_bottlenecks

        self.readout = args.readout
        self.pos_enc = args.pos_enc
        self.pos_enc_dim = args.pos_enc_dim
        self.lappe = args.lappe
        self.lap_dim = args.lap_dim
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward is not None else 2048

        self.Bottlnecks = nn.Parameter(torch.randn(1, self.num_bottlenecks, self.hidden_dim))
        
        self.node_embedding_sc = nn.Linear(self.sc_features, self.hidden_dim)
        self.node_embedding_fc = nn.Linear(self.fc_features, self.hidden_dim)

        # self.embedding_sc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        # self.embedding_fc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )

        self.sc_subGraph_convs = nn.ModuleList()
        self.fc_subGraph_convs = nn.ModuleList()
        self.fusion_transformer_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            # self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            # self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            # self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            self.fusion_transformer_convs.append(nn.TransformerEncoderLayer(
                self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
        self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
        self.fusion_transformer_convs.append(nn.TransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
    
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data):# h_lap_pos_enc=None):
        batch_size = data.batch.max().item() + 1
        
        sc_x, sc_sub_x, sc_edge_index, sc_sub_x_index, sc_sub_edge_idx, sc_node_to_subgraph, sc_subgraph_to_graph = self.process(data=data, modality='sc')
        fc_x, fc_sub_x, fc_edge_index, fc_sub_x_index, fc_sub_edge_idx, fc_node_to_subgraph, fc_subgraph_to_graph = self.process(data=data, modality='fc')
        
        batch = torch.tensor([[i] * 90 for i in range(batch_size)]).view(-1)
        if self.use_cuda:
            batch = batch.cuda()
        
        multi_scale_bottlenecks = []

        for i in range(self.num_layers-1):
            sc_x = self.sc_subGraph_convs[i](sc_sub_x, sc_sub_edge_idx, sc_node_to_subgraph)
            fc_x = self.fc_subGraph_convs[i](fc_sub_x, fc_sub_edge_idx, fc_node_to_subgraph)
    
            sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            
            fusion_x = torch.cat((sc_x, fc_x), dim=1)
            
            fusion_x = self.fusion_transformer_convs[i](fusion_x)
            
            sc_x = fusion_x[:,:90,:]
            sc_x = fusion_x[:,90:,:]
            
            sc_x = sc_x.contiguous().view(-1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(-1, self.hidden_dim)

            
            sc_x, fc_x = self.bns[i](sc_x), self.bns[i](fc_x)
            sc_x, fc_x = F.relu(sc_x), F.relu(fc_x)
            sc_x, fc_x = F.dropout(sc_x, p=self.dropout, training=self.training), F.dropout(fc_x, p=self.dropout, training=self.training)
            
            # pooled_bottlenecks = global_mean_pool(b_bottlenecks, batch)
            # multi_scale_bottlenecks.append(pooled_bottlenecks)
            
            # b_bottlenecks = b_bottlenecks.contiguous().view(batch_size, -1, self.hidden_dim)

            sc_sub_x = sc_x[sc_sub_x_index] #get_subx(x, sub_x_idx, subgraph_to_graph, data.batch)
            fc_sub_x = fc_x[fc_sub_x_index]
        
        sc_x = self.sc_subGraph_convs[-1](sc_sub_x, sc_sub_edge_idx, sc_node_to_subgraph)
        fc_x = self.fc_subGraph_convs[-1](fc_sub_x, fc_sub_edge_idx, fc_node_to_subgraph)
 
        sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        
        fusion_x = torch.cat((sc_x, fc_x), dim=1)
        
        fusion_x = self.fusion_transformer_convs[i](fusion_x)
        
        sc_x = fusion_x[:,:90,:]
        sc_x = fusion_x[:,90:,:]
        
        sc_x = sc_x.contiguous().view(-1, self.hidden_dim)
        fc_x = fc_x.contiguous().view(-1, self.hidden_dim)
        
        pooled_sc = global_mean_pool(sc_x, batch)
        pooled_fc = global_mean_pool(fc_x, batch)
        
        pooled_sc = pooled_sc.unsqueeze(1)
        pooled_fc = pooled_fc.unsqueeze(1)
        
        # x = self.bns[-1](x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        
        ## 最后一个b_bottlenecks做classifier
        logit_sc = self.classifier(pooled_sc)
        logit_fc = self.classifier(pooled_fc)
        logit = (logit_sc + logit_fc)/2
        logit = torch.squeeze(logit, dim=1)
        
        # ## 多尺度b_bottlenecks做classifier
        # # 1. list --> tensor --> view
        # multi_scale_bottlenecks = torch.stack(multi_scale_bottlenecks, dim=0).permute(1,0,2)
        # # 2. 自注意力计算
        # multi_scale_bottlenecks, attn_output_weights  = self.multi_scale_fuse(multi_scale_bottlenecks, multi_scale_bottlenecks, multi_scale_bottlenecks)
        # # 3. 平均
        # multi_scale_bottlenecks = torch.mean(multi_scale_bottlenecks, dim=1).unsqueeze(1)
        # # 4. 分类
        # logit = self.classifier(multi_scale_bottlenecks).squeeze(1)
        
        return logit, sc_x, fc_x    # (logit: bz * label_num;) (sc_x, fc_x: bz * node_num * hidden_dim)
    
    def process(self, data, modality='sc'):
        edge_index, x = data.__getattr__('{}_original_edge_index'.format(modality)),  data.__getattr__('{}_original_x'.format(modality))
        sub_x_index, sub_edge_idx = data.__getattr__('{}_node_index'.format(modality)), data.__getattr__('{}_edge_index'.format(modality)) 
        node_to_subgraph, subgraph_to_graph = data.__getattr__('{}_node_to_subgraph'.format(modality)), data.__getattr__('{}_subgraph_to_graph'.format(modality)) 

        x = eval('self.node_embedding_{}'.format(modality))(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        sub_x = x[sub_x_index] 
        
        sub_x = F.dropout(sub_x, p= self.dropout, training=self.training)
            
        return x, sub_x, edge_index, sub_x_index, sub_edge_idx, node_to_subgraph, subgraph_to_graph

        
class GNN_Bottlnecks(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_cuda = args.use_cuda
        
        self.num_heads = args.num_heads
        self.out_dim = args.hidden_dim
        self.sc_features = args.sc_features
        self.fc_features = args.fc_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_bottlenecks = args.num_bottlenecks

        self.readout = args.readout
        self.pos_enc = args.pos_enc
        self.pos_enc_dim = args.pos_enc_dim
        self.lappe = args.lappe
        self.lap_dim = args.lap_dim
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.residual = args.residual
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward is not None else 2048

        self.Bottlnecks = nn.Parameter(torch.randn(1, self.num_bottlenecks, self.hidden_dim))
        
        self.node_embedding_sc = nn.Linear(self.sc_features, self.hidden_dim)
        self.node_embedding_fc = nn.Linear(self.fc_features, self.hidden_dim)

        # self.embedding_sc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        # self.embedding_fc_lap_pos_enc = nn.Linear(self.lap_dim, self.hidden_dim)
        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )

        self.sc_subGraph_convs = nn.ModuleList()
        self.fc_subGraph_convs = nn.ModuleList()
        self.fusion_transformer_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            # self.sc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            # self.fc_subGraph_convs.append(GINSublayer(self.hidden_dim, self.hidden_dim, sublayers=1, subhiddens=self.hidden_dim))
            # self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            self.sc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
            self.fc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
            self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
            
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.sc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
        self.fc_subGraph_convs.append(GINConv(mlp(inchannel=self.hidden_dim, hidden=self.hidden_dim, outchannel=self.hidden_dim),train_eps=True))
        self.fusion_transformer_convs.append(FusionTransformerEncoderLayer(self.hidden_dim, self.num_heads, self.dim_feedforward, self.dropout, batch_first=True))
    
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data):# h_lap_pos_enc=None):
        batch_size = data.batch.max().item() + 1
        
        sc_x, sc_sub_x, sc_edge_index, sc_sub_x_index, sc_sub_edge_idx, sc_node_to_subgraph, sc_subgraph_to_graph = self.process(data=data, modality='sc')
        fc_x, fc_sub_x, fc_edge_index, fc_sub_x_index, fc_sub_edge_idx, fc_node_to_subgraph, fc_subgraph_to_graph = self.process(data=data, modality='fc')
        
        batch = torch.tensor([[i] * self.num_bottlenecks for i in range(batch_size)]).view(-1)
        if self.use_cuda:
            batch = batch.cuda()
        
        multi_scale_bottlenecks = []

        for i in range(self.num_layers-1):
            sc_x = self.sc_subGraph_convs[i](sc_x, sc_edge_index)
            fc_x = self.fc_subGraph_convs[i](fc_x, fc_edge_index)
            
            sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
            
            sc_x, fc_x, b_bottlenecks = self.fusion_transformer_convs[i](sc_x, fc_x, b_bottlenecks)
            
            sc_x = sc_x.contiguous().view(-1, self.hidden_dim)
            fc_x = fc_x.contiguous().view(-1, self.hidden_dim)
            b_bottlenecks = b_bottlenecks.contiguous().view(-1, self.hidden_dim)
            
            sc_x, fc_x, b_bottlenecks = self.bns[i](sc_x), self.bns[i](fc_x), self.bns[i](b_bottlenecks)
            sc_x, fc_x, b_bottlenecks = F.relu(sc_x), F.relu(fc_x), F.relu(b_bottlenecks)
            sc_x, fc_x, b_bottlenecks = F.dropout(sc_x, p=self.dropout, training=self.training), F.dropout(fc_x, p=self.dropout, training=self.training), F.dropout(b_bottlenecks, p=self.dropout, training=self.training)

            pooled_bottlenecks = global_mean_pool(b_bottlenecks, batch)
            multi_scale_bottlenecks.append(pooled_bottlenecks)
            
            b_bottlenecks = b_bottlenecks.contiguous().view(batch_size, -1, self.hidden_dim)

        
        sc_x = self.sc_subGraph_convs[-1](sc_x, sc_edge_index)
        fc_x = self.fc_subGraph_convs[-1](fc_x, fc_edge_index)
        
        sc_x = sc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        fc_x = fc_x.contiguous().view(batch_size, -1, self.hidden_dim)
        
        sc_x, fc_x, b_bottlenecks = self.fusion_transformer_convs[-1](sc_x, fc_x, b_bottlenecks)

        sc_x, fc_x, b_bottlenecks = sc_x.contiguous(), fc_x.contiguous(), b_bottlenecks.contiguous()
        
        b_bottlenecks = b_bottlenecks.contiguous().view(-1, self.hidden_dim)
        pooled_bottlenecks = global_mean_pool(b_bottlenecks, batch)
        
        multi_scale_bottlenecks.append(pooled_bottlenecks)

        ## 最后一个b_bottlenecks做classifier
        logit = self.classifier(multi_scale_bottlenecks[-1].unsqueeze(1))
        logit = torch.squeeze(logit, dim=1)
        
        
        return logit, sc_x, fc_x    # (logit: bz * label_num;) (sc_x, fc_x: bz * node_num * hidden_dim)
    
    def process(self, data, modality='sc'):
        edge_index, x = data.__getattr__('{}_original_edge_index'.format(modality)),  data.__getattr__('{}_original_x'.format(modality))
        sub_x_index, sub_edge_idx = data.__getattr__('{}_node_index'.format(modality)), data.__getattr__('{}_edge_index'.format(modality)) 
        node_to_subgraph, subgraph_to_graph = data.__getattr__('{}_node_to_subgraph'.format(modality)), data.__getattr__('{}_subgraph_to_graph'.format(modality)) 

        x = eval('self.node_embedding_{}'.format(modality))(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        sub_x = x[sub_x_index] 
        
        sub_x = F.dropout(sub_x, p= self.dropout, training=self.training)
            
        return x, sub_x, edge_index, sub_x_index, sub_edge_idx, node_to_subgraph, subgraph_to_graph