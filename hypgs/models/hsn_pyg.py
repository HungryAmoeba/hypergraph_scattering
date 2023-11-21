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
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, GlobalAttention

class HyperDiffusion(MessagePassing):
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

class HyperScatteringModule(nn.Module):
    def __init__(self, in_channels, trainable_laziness=False, trainable_scales=False, activation="blis", fixed_weights=True, normalize="right", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.device = device
        self.diffusion_layer1 = HyperDiffusion(in_channels, in_channels, trainable_laziness, fixed_weights, normalize)
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=torch.float, device=self.device, requires_grad=trainable_scales))
        if activation == "blis":
            self.activations = [lambda x: torch.relu(x), lambda x: torch.relu(-x)]
        elif activation == None:
            self.activations = [lambda x : x]
        elif activation == "modulus":
            self.activations = [lambda x: torch.abs(x)]
        elif activation == "leaky_relu":
            m = nn.LeakyReLU()
            self.activations = [lambda x: m(x)]

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None):
        features = x.shape[1]
        node_features = [x]
        edge_features = [hyperedge_attr]
        for i in range(16):
            node_feat, edge_feat = self.diffusion_layer1(x=node_features[-1], hyperedge_index=hyperedge_index, hyperedge_weight=hyperedge_weight, hyperedge_attr=edge_features[-1])
            node_features.append(node_feat)
            edge_features.append(edge_feat)
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = rearrange(node_features, 'i j k -> i j k')
        edge_diffusion_levels = rearrange(edge_features, 'i j k -> i j k')
        wavelet_coeffs = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        wavelet_coeffs_edges = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, edge_diffusion_levels)
        # TODO add batch norm here!
        activated = [self.activations[i](wavelet_coeffs) for i in range(len(self.activations))]
        activated_edges = [self.activations[i](wavelet_coeffs_edges) for i in range(len(self.activations))]
        s_nodes = rearrange(activated, 'a w n f -> n (w f a)')
        s_edges = rearrange(activated_edges, 'a w e f -> e (w f a)')
        return s_nodes, s_edges
    
    def out_features(self):
        return 6 * self.in_channels * len(self.activations)

class HSN(nn.Module):
    def __init__(self, 
                in_channels, 
                hidden_channels, 
                out_channels, 
                trainable_laziness = False, 
                trainable_scales = False, 
                activation = "modulus", 
                fixed_weights=True, 
                layout = ['hsm','hsm'], 
                normalize="right",
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                pooling=None,
                **kwargs):
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.trainable_laziness = trainable_laziness 
        self.trainable_scales = trainable_scales 
        self.activation = activation 
        self.fixed_weights = fixed_weights
        self.layout = layout 
        self.layers = []
        self.out_dimensions = [in_channels]
        self.normalize = normalize
        self.device = device
        self.pooling = pooling
        if pooling == 'attention':
            raise NotImplementedError
            # gate_nn = torch.nn.Sequential(
            #     torch.nn.Linear(hidden_channels, 1),
            #     torch.nn.Sigmoid()
            # )
            # self.attention_pool = GlobalAttention(gate_nn=gate_nn)


        for layout_ in layout:
            if layout_ == 'hsm':
                self.layers.append(HyperScatteringModule(self.out_dimensions[-1], 
                                                            trainable_laziness = trainable_laziness,
                                                            trainable_scales = self.trainable_scales, 
                                                            activation = self.activation, 
                                                            fixed_weights=self.fixed_weights,
                                                            normalize=normalize,
                                                            device=kwargs.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))))
                self.out_dimensions.append(self.layers[-1].out_features() )
            elif layout_ == 'dim_reduction':
                input_dim = self.out_dimensions[-1]
                output_dim = input_dim//2
                self.out_dimensions.append(output_dim)
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                raise ValueError("Not yet implemented")
             
        self.layers = nn.ModuleList(self.layers)

        # currently share backend MLPs for the node and edge features
        self.batch_norm = BatchNorm(self.out_dimensions[-1])

        self.fc1 = Linear(self.out_dimensions[-1], self.out_dimensions[-1]//2)
        self.fc2 = nn.Linear(self.out_dimensions[-1]//2, self.out_channels)
        #self.fc3 = nn.Linear(128, 64)
        #self.fc4 = nn.Linear(64, self.out_channels)
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_dimensions[-1]//2)
        #self.batch_norm2 = nn.BatchNorm1d(128)
        #self.batch_norm3 = nn.BatchNorm1d(64)

        self.mlp = nn.Sequential(
            self.fc1,
            self.batch_norm1,
            self.relu,
            self.fc2
        )


    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None,
                batch: Optional[torch.Tensor] = None):
        
        for il, layer in enumerate(self.layers):
            if self.layout[il] == 'hsm':
                x, hyperedge_attr = layer(x, hyperedge_index, hyperedge_weight, hyperedge_attr, num_edges)
                # TODO add batch norm before non-linearity inside the hsm!
            elif self.layout[il] == 'dim_reduction':
                x = layer(x) # TODO add batch norm and non-linearity!
                hyperedge_attr = layer(hyperedge_attr) 
            else:
                raise ValueError
        # Apply selected pooling
        if self.pooling is not None:
            assert batch is not None
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'attention':
            raise NotImplementedError
            # x = self.attention_pool(x, batch)

        x = self.batch_norm(x)

        x = self.mlp(x)

        # compute the same process on the edges:
        hyperedge_attr = self.batch_norm(hyperedge_attr)
        hyperedge_attr = self.mlp(hyperedge_attr)

        return x, hyperedge_attr
