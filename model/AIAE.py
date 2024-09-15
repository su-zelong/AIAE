import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from .model_utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Lower_encoder(nn.Module):
    def __init__(self, args, in_channel, out_channel):
        super(Lower_encoder, self).__init__()
        self.args = args
        self.gcn1 = GCNConv(in_channels=in_channel, out_channels=out_channel)
        self.gcn2 = GCNConv(in_channels=out_channel, out_channels=out_channel)
        self.lower_norm = nn.LayerNorm(out_channel)
    
    def forward(self, x, train_edges):
        h1_0 = F.relu(self.lower_norm(self.gcn1(x, train_edges)))
        # h1_0 = self.lower_norm(h1_0)
        h2_0 = F.relu(self.lower_norm(self.gcn2(h1_0, train_edges)))
        # h2_0 = self.lower_norm(h2_0)
        return h1_0, h2_0

class Upper_encoder(nn.Module):
    def __init__(self, args, in_channel, out_channel):
        super(Upper_encoder, self).__init__()
        self.args = args
        self.in_channel = in_channel
        self.gcn3 = GCNConv(in_channels=in_channel, out_channels=out_channel)
        self.gcn4 = GCNConv(in_channels=out_channel, out_channels=out_channel)
        self.upper_norm = nn.LayerNorm(out_channel)

    def forward(self, x, all_edges):
        '''n = all nodes number'''
        # idt = torch.eye(int((self.in_channel)))
        z_1 = F.relu(self.upper_norm(self.gcn3(x, all_edges)))
        # z_1 = self.upper_norm(z_1)
        z_2 = F.relu(self.upper_norm(self.gcn4(z_1, all_edges)))
        # z_2 = self.upper_norm(z_2)
        return z_1, z_2

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.embed = nn.Embedding(num_embeddings=args.node_length, embedding_dim=64)
        self.lower_encoder = Lower_encoder(args, in_channel=64, out_channel=64)
        self.upper_encoder = Upper_encoder(args, in_channel=64, out_channel=64)

    def forward(self, train_x, train_edges, train_mask, all_edges):
        '''train_x:648*3703; train_edges:2*42; all_edges: 2*2204'''
        node_idx = torch.arange(self.args.node_length)    # 直接对全体节点进行归一化
        embed = self.embed(node_idx)    # 1620*64    两层GCN不在一个量级，该归一化归一化
        o_embed = embed[train_mask]    # 684*64    observed node embed
        # lower encoder
        h1_0, h2_0 = self.lower_encoder(o_embed, train_edges)   # 2*42
        # upper encoder
        z_1, z_2 = self.upper_encoder(embed, all_edges)
        return h1_0, h2_0, z_1, z_2


class Decoder_imputation(nn.Module):
    def __init__(self, args) -> None:
        super(Decoder_imputation, self).__init__()
        self.args = args
        self.mlp = nn.Linear(in_features=64, out_features=64) 
    
    def forward(self, z_2, z_2_):
        x_ = self.mlp(z_2)    # 1620*64
        adj_ = F.sigmoid(torch.matmul(z_2, torch.transpose(z_2_, -1, -2)))   # 1620*1620
        return x_, adj_


class Decoder_MS(nn.Module):
    def __init__(self, args) -> None:
        super(Decoder_MS, self).__init__()
        self.args = args
        self.gcn5 = GCNConv(in_channels=64, out_channels=64)
        self.ms_norm = nn.LayerNorm(64)
    
    def forward(self, z_1_, z_2_, train_mask, edges):
        # z_ = torch.cat((z_1_, z_2_), dim=-1)
        z_ = z_1_ + z_2_
        r_idx = torch.where(train_mask == False)[0]
        num_random_mask = int(self.args.missing_length * self.args.random_mask_rate)
        mask_idx = r_idx[torch.randperm(self.args.missing_length)[num_random_mask:]]    # 随机选择这些位置不mask
        z__ = torch.rand_like(z_)
        z__[mask_idx] = z_[mask_idx]
        output = F.relu(self.ms_norm(self.gcn5(z_, edges))) + F.relu(self.ms_norm(self.gcn5(z__, edges)))
        return output
        