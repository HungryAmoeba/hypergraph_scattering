"""
HSN rewritten with pytorch geometric, can operate on batched hypergraphs. 
the data is stored in the format of pytorch geometric.
see https://github.com/pyg-team/pytorch_geometric/blob/cf24b4bcb4e825537ba08d8fc5f31073e2cd84c7/torch_geometric/data/hypergraph_data.py
for example: 
    hyperedge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3],
        [0, 0, 0, 1, 1, 1],
    ])
    hyperedge_weight = torch.tensor([1, 1], dtype=torch.float)

modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hypergraph_conv.html#HypergraphConv
"""
import dhg 
import torch
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange
from torch.nn import Linear 
from torch_geometric.nn.pool import global_mean_pool 
from torch_geometric.nn import GCNConv 
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from .hyper_scattering_net import LazyLayer
from torch_geometric.utils import scatter, softmax

class HGDiffsion(MessagePassing):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            trainable_laziness=False,
            fixed_weights=True,
            normalize="right",
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        assert normalize in ["right", "left", "symmetric"], f"normalize must be one of 'right', 'left', or 'symmetric', not {self.normalize}"

        self.normalize = normalize
 
        # in the future, we could make this time independent, but spatially dependent, as in GRAND
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        # in the future, I'd like to have different weights based on the hypergraph edge size
        if not self.fixed_weights:
            self.lin_self = torch.nn.Linear(in_channels, out_channels)
            self.lin_neigh = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        
        # this is the degree of the vertices (taken inverse)
        D_v_inv = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D_v_inv = 1.0 / D_v_inv
        D_v_inv[D_v_inv == float("inf")] = 0
        # this is the degree of the hyperedges (taken inverse)
        D_he_inv = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        D_he_inv = 1.0 / D_he_inv
        D_he_inv[D_he_inv == float("inf")] = 0
    
        if self.normalize == "left":
            out_edge = self.propagate(hyperedge_index, x=x, norm=D_he_inv,
                                size=(num_nodes, num_edges))
            out_edge = self.laziness_weight_process_edge(out_edge, hyperedge_attr)
            out_node = self.propagate(hyperedge_index.flip([0]), x=out_edge, norm=D_v_inv,
                                size=(num_edges, num_nodes))
            out_node = self.laziness_weight_process_node(out_node, x)
        elif self.normalize == "right":
            out = D_v_inv.view(-1, 1) * x
            out_edge = self.propagate(hyperedge_index, x=out,
                                size=(num_nodes, num_edges))
            out_edge = self.laziness_weight_process_edge(out_edge, hyperedge_attr)
            out = D_he_inv.view(-1, 1) * out_edge
            out_node = self.propagate(hyperedge_index.flip([0]), x=out,
                                size=(num_edges, num_nodes))
            out_node = self.laziness_weight_process_node(out_node, x)
        elif self.normalize == "symmetric":
            D_v_inv_sqrt = D_v_inv.sqrt()
            out = D_v_inv_sqrt.view(-1, 1) * x
            out_edge = self.propagate(hyperedge_index, x=out, norm=D_he_inv,
                                size=(num_nodes, num_edges))
            out_edge = self.laziness_weight_process_edge(out_edge, hyperedge_attr)
            out_node = self.propagate(hyperedge_index.flip([0]), x=out_edge, norm=D_v_inv_sqrt,
                                size=(num_edges, num_nodes))
            out_node = self.laziness_weight_process_node(out_node, x)
        else: 
            raise ValueError(f"normalize must be one of 'right', 'left', or 'symmetric', not {self.normalize}")
        return out_node, out_edge
    
    def message(self, x_j: torch.Tensor, norm_i: Optional[torch.Tensor] = None) -> torch.Tensor:
        if norm_i is None:
            out = x_j
        else:
            out = norm_i.view(-1, 1) * x_j
        return out
    
    def laziness_weight_process_edge(self, out_edge, hyperedge_attr):
        if not self.fixed_weights:
            out_edge = self.lin_neigh(out_edge) 
            hyperedge_attr = self.lin_self(out_edge)
        if self.trainable_laziness and hyperedge_attr is not None:
            out_edge = self.lazy_layer(out_edge, hyperedge_attr)
        return out_edge
    
    def laziness_weight_process_node(self, out_node, x):
        if not self.fixed_weights:
            out_node = self.lin_neigh(out_node)
            x = self.lin_self(x)
        if self.trainable_laziness:
            out_node = self.lazy_layer(out_node, x)
        return out_node

# class HSNBatch(HSN):
#     def __init__(self, 
#                  in_channels, 
#                  hidden_channels, 
#                  out_channels, 
#                  trainable_laziness = False, 
#                  trainable_scales = False, 
#                  activation = "modulus", 
#                  fixed_weights=True, 
#                  layout = ['hsm','hsm'], 
#                  pooling = 'mean',
#                  **kwargs):
        
#         super().__init__(in_channels, hidden_channels, out_channels, trainable_laziness, trainable_scales, activation, fixed_weights, layout, **kwargs)
#         self.pooling = pooling


#     def forward(self, data):
#         x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
        
#         NotImplemented
#     # def forward(self, hg: dhg.Hypergraph,  X: torch.Tensor, Y: torch.Tensor):
#     #     for il, layer in enumerate(self.layers):
#     #         if self.layout[il] == 'hsm':
#     #             X, Y = layer(hg, X, Y)
#     #         elif self.layout[il] == 'dim_reduction':
#     #             X = layer(X)
#     #             Y = layer(Y) 
#     #         else:
#     #             X, Y = layer(hg, X, Y)
#     #     #import pdb; pdb.set_trace()
#     #     X = self.batch_norm(X)
#     #     X = self.mlp(X)
#     #     # X = self.lin1(X)
#     #     # X = self.act(X)
#     #     # X = self.lin2(X)
#     #     # X = self.act(X)
#     #     # X = self.lin3(X)

#     #     # compute the same process on the edges:
#     #     Y = self.batch_norm(Y)
#     #     Y = self.mlp(Y)
#     #     # Y = self.lin1(Y)
#     #     # Y = self.act(Y)
#     #     # Y = self.lin2(Y)
#     #     # Y = self.act(Y)
#     #     # Y = self.lin3(Y)

#     #     return X,Y