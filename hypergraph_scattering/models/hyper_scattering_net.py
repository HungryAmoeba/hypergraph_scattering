import dhg 
import torch
import torch.nn as nn
from typing import Tuple, Optional

class LazyLayer(torch.nn.Module):
    
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
    

class HyperDiffusion(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            trainable_laziness=False,
            fixed_weights=True
    ):
        super().__init__()
        self.trainable_laziness= trainable_laziness 
        self.fixed_weights = fixed_weights 
        # in the future, we could make this time independent, but spatially dependent, as in GRAND
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        # in the future, I'd like to have different weights based on the hypergraph edge size
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, X: torch.Tensor, hg: dhg.Hypergraph, Y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            if not self.fixed_weights:
                X = self.lin(X)
            
            # X has shape num_nodes, num_features

            # propagate from nodes to hyperedges 
            inv_deg_v = hg.D_v_neg_1.values()
            inv_deg_v = torch.nan_to_num(inv_deg_v)
            # I should degree normalize first
            X_norm = torch.einsum('ij,i->ij', X, inv_deg_v)

            edge_feat = dhg.v2e(X_norm, aggr = 'add')
            if self.trainable_laziness and Y is not None:
                edge_feat = self.lazy_layer(edge_feat, Y)

            # propagate back from hyperedges to nodes 
            inv_deg_e = hg.D_e_neg_1.values()
            inv_deg_e = torch.nan_to_num(inv_deg_e)
            edge_feat_norm = torch.einsum('ij,i->ij', edge_feat, inv_deg_e)

            node_feat = dhg.v2e(edge_feat_norm, agg = 'add')
            
            return node_feat, edge_feat 

        



        
