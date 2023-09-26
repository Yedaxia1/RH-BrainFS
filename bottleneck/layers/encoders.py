import torch
from torch import nn

class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, degree=None, mask=None, src_key_padding_mask=None, JK=False):
        output = src
        xs = []
        for mod in self.layers:
            output = mod(output, pe=pe, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
            xs.append(output)
        if self.norm is not None:
            output = self.norm(output)
        if JK:
            output = torch.cat(xs,-1)

        return output

class FusionTransformerEncoder(nn.TransformerEncoder):
    def forward(self, sc_feature, fc_feature, bottlenecks):

        sc_feature_res, fc_feature_res, bottlenecks_res = sc_feature, fc_feature, bottlenecks
        for mod in self.layers:
            sc_feature_res, fc_feature_res, bottlenecks_res = mod(sc_feature_res, fc_feature_res, bottlenecks_res)

        return sc_feature_res, fc_feature_res, bottlenecks_res
    
    
        
        