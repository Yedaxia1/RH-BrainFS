
import warnings
import torch
import torch.nn as nn
import torch_geometric.nn as tnn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, global_max_pool as gxp

def diff_multi_head_attention_forward(query,
                                      key,
                                      value,
                                      pe,
                                      embed_dim_to_check,
                                      num_heads,
                                      in_proj_weight,
                                      in_proj_bias,
                                      bias_k,
                                      bias_v,
                                      add_zero_attn,
                                      dropout_p,
                                      out_proj_weight,
                                      out_proj_bias,
                                      training=True,
                                      key_padding_mask=None,
                                      need_weights=True,
                                      attn_mask=None,
                                      use_separate_proj_weight=False,
                                      q_proj_weight=None,
                                      k_proj_weight=None,
                                      v_proj_weight=None,
                                      static_k=None,
                                      static_v=None
                                      ):

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by \
            num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = nn.functional.linear(query, in_proj_weight,
                                           in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and
                # in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = nn.functional.linear(query, q_proj_weight_non_opt,
                                     in_proj_bias[0:embed_dim])
            k = nn.functional.linear(key, k_proj_weight_non_opt,
                                     in_proj_bias[embed_dim:(embed_dim * 2)])
            v = nn.functional.linear(value, v_proj_weight_non_opt,
                                     in_proj_bias[(embed_dim * 2):])
        else:
            q = nn.functional.linear(query, q_proj_weight_non_opt,
                                     in_proj_bias)
            k = nn.functional.linear(key, k_proj_weight_non_opt,
                                     in_proj_bias)
            v = nn.functional.linear(value, v_proj_weight_non_opt,
                                     in_proj_bias)
    k = q
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)],
                                      dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((
                            key_padding_mask.size(0), 1),
                            dtype=key_padding_mask.dtype,
                            device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:],
                       dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:],
                       dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros(
                    (attn_mask.size(0), 1),
                    dtype=attn_mask.dtype,
                    device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros(
                        (key_padding_mask.size(0), 1),
                        dtype=key_padding_mask.dtype,
                        device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len,
                                                src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads,
                                                       tgt_len, src_len)

    if pe is not None:
        pe = torch.repeat_interleave(pe, repeats=num_heads, dim=0)
    # numerical stability
    max_val = attn_output_weights.max(dim=-1, keepdim=True)[0]
    attn_output_weights = torch.exp(attn_output_weights - max_val)
    attn_output_weights_temp = torch.zeros_like(attn_output_weights)
    attn_output_weights_temp[:] = attn_output_weights[:]
    if pe is not None:
        attn_output_weights_temp[:,1:,1:] = attn_output_weights[:,1:,1:] * pe
    attn_output_weights = attn_output_weights_temp
    attn_output_weights = attn_output_weights / attn_output_weights.sum(
        dim=-1, keepdim=True).clamp(min=1e-6)
    attn_output_weights = nn.functional.dropout(attn_output_weights,
                                                p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight,
                                       out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        # return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None

class DiffMultiheadAttention(nn.modules.activation.MultiheadAttention):
    def forward(self, query, key, value, pe, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        if hasattr(
                self, '_qkv_same_embed_dim'
                ) and self._qkv_same_embed_dim is False:
            return diff_multi_head_attention_forward(
                    query, key, value, pe, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.bias_k,
                    self.bias_v, self.add_zero_attn, self.dropout,
                    self.out_proj.weight, self.out_proj.bias,
                    training=self.training, key_padding_mask=key_padding_mask,
                    need_weights=need_weights, attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttentio, module has benn implemented. \
                        Please re-train your model with the new module',
                              UserWarning)
            return diff_multi_head_attention_forward(
                    query, key, value, pe, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.bias_k,
                    self.bias_v, self.add_zero_attn, self.dropout,
                    self.out_proj.weight, self.out_proj.bias,
                    training=self.training, key_padding_mask=key_padding_mask,
                    need_weights=need_weights, attn_mask=attn_mask)


class DiffTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_norm=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = DiffMultiheadAttention(d_model, nhead,
                                                dropout=dropout, bias=False)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.scaling = None

    def forward(self, src, pe, degree=None, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, pe, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        if degree is not None:
            src2 = degree.transpose(0, 1).contiguous().unsqueeze(-1) * src2
        else:
            if self.scaling is None:
                self.scaling = 1. / pe.diagonal(dim1=1, dim2=2).max().item()
            src_temp = torch.zeros_like(src2)
            src_temp[:] = src2[:]
            src_temp[1:,:,:] = (self.scaling * pe.diagonal(dim1=1, dim2=2)).transpose(0, 1).contiguous().unsqueeze(-1) * src2[1:,:,:]
            src2 = src_temp
        src = src + self.dropout1(src2)
        if self.batch_norm:
            bsz = src.shape[1]
            src = src.view(-1, src.shape[-1])
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.batch_norm:
            src = src.view(-1, bsz, src.shape[-1])
        return src

class FusionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_norm=False, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

        self.hidden_dim = d_model
        self.num_heads = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first
        self.batch_norm = batch_norm
        
        self.sc_fusion_encoder_layer = nn.TransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, dropout, batch_first=self.batch_first)
        self.fc_fusion_encoder_layer = nn.TransformerEncoderLayer(
            self.hidden_dim, self.num_heads, self.dim_feedforward, dropout, batch_first=self.batch_first)


    def forward(self, sc_feature, fc_feature, bottlenecks):
        b_sc, n_sc, _ = sc_feature.shape
        b_fc, n_fc, _ = fc_feature.shape

        sc_feature_temp = torch.cat((sc_feature, bottlenecks), dim=1)
        fc_feature_temp = torch.cat((fc_feature, bottlenecks), dim=1)

        sc_feature_temp = self.sc_fusion_encoder_layer(sc_feature_temp)
        fc_feature_temp = self.fc_fusion_encoder_layer(fc_feature_temp)

        sc_feature_res = sc_feature_temp[:,:n_sc,:]
        fc_feature_res = fc_feature_temp[:,:n_fc,:]

        bottlenecks_res = (sc_feature_temp[:,n_sc:,:] + fc_feature_temp[:,n_fc:,:]) / 2

        return sc_feature_res, fc_feature_res, bottlenecks_res


class GCNSublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(GCNSublayer, self).__init__()
        self.sublayers = sublayers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(sublayers-1):
            self.sub_gnns.append(tnn.GCNConv(in_channels=in_channels, out_channels=subhiddens))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens

        self.sub_gnns.append(tnn.GCNConv(in_channels=in_channels, out_channels=out_channels))
        # self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        xs = []
        for i in range(self.sublayers - 1):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5,self.training)
            xs.append(x)
        x = self.sub_gnns[-1](x, sub_edge_index)
        x = F.relu(x)
        xs.append(x)
        x = gmp(torch.cat(xs,dim=-1), node_to_subgraph)
        return x
    
class GINSublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(GINSublayer, self).__init__()
        self.sublayers = sublayers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )


        for _ in range(sublayers-1):
            self.sub_gnns.append(tnn.GINConv(mlp(inchannel=in_channels, hidden=subhiddens, outchannel=subhiddens),train_eps=True))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens

        self.sub_gnns.append(tnn.GINConv(mlp(inchannel=in_channels, hidden=subhiddens, outchannel=out_channels),train_eps=True))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        xs = []
        for i in range(self.sublayers):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5,self.training)
            xs.append(x)
        # x = self.sub_gnns[-1](x, sub_edge_index)
        # x = F.relu(x)
        # xs.append(x)
        # x = torch.cat([gxp(torch.cat(xs,dim=-1), node_to_subgraph),gmp(torch.cat(xs,dim=-1), node_to_subgraph)], dim=-1)
        # x = torch.cat([gxp(torch.cat(xs,dim=-1), node_to_subgraph),gmp(torch.cat(xs,dim=-1), node_to_subgraph)], dim=-1)
        x = gmp(torch.cat(xs,dim=-1), node_to_subgraph)
        # x = gmp(xs[-1], node_to_subgraph)
        return x

class GINSublayer_VN(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(GINSublayer_VN, self).__init__()
        self.sublayers = sublayers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )
        self.vn_ebd = torch.nn.Embedding(1, subhiddens)
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for _ in range(sublayers-1):
            self.sub_gnns.append(tnn.GINConv(mlp(inchannel=in_channels, hidden=subhiddens, outchannel=subhiddens),train_eps=True))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(subhiddens, 2*subhiddens),torch.nn.BatchNorm1d(2*subhiddens), torch.nn.ReLU(),
                torch.nn.Linear(2*subhiddens, subhiddens), torch.nn.BatchNorm1d(subhiddens), torch.nn.ReLU()))

        self.sub_gnns.append(tnn.GCNConv(in_channels=in_channels, out_channels=out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.vn_ebd.weight.data, 0)
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        vne = self.vn_ebd(torch.zeros(node_to_subgraph[-1].item() + 1, dtype=sub_edge_index.dtype, device=sub_edge_index.device))
        # x = x + vne[node_to_subgraph]
        xs = []
        for i in range(self.sublayers-1):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5,self.training)
            xs.append(x)

            vnet = gap(xs[i], node_to_subgraph) + vne
            vne = F.dropout(self.mlp_virtualnode_list[i](vnet), 0.5, training = self.training)
            x = x + vne[node_to_subgraph]
        
        xs[-1] = xs[-1] + vne[node_to_subgraph]
        x = F.dropout(self.bns[-1](self.sub_gnns[-1](xs[-1], sub_edge_index)), 0.5,self.training)
        xs.append(x)

        node_p = 0
        for layer in range(self.sublayers):
            node_p += xs[layer]
        x = gmp(node_p, node_to_subgraph)
        return x

class MLPSublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(MLPSublayer, self).__init__()
        self.sublayers = sublayers
        self.sub_lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(sublayers-1):
            self.sub_lins.append(nn.Linear(in_channels, subhiddens))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens

        self.sub_lins.append(nn.Linear(in_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_lins:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        xs = []
        for i in range(self.sublayers):
            x = self.sub_lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5,self.training)
            xs.append(x)
        
        x = gmp(torch.cat(xs,dim=-1), node_to_subgraph)
        return x

