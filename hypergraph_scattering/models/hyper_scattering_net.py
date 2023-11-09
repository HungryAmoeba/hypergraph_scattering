import dhg 
import torch
import torch.nn as nn
from typing import Tuple, Optional

# a good way to make hyper edges:
# dhg.Hypergraph.from_graph_kHop()

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

        edge_feat = hg.v2e(X_norm, aggr = 'sum')
        if self.trainable_laziness and Y is not None:
            edge_feat = self.lazy_layer(edge_feat, Y)

        # propagate back from hyperedges to nodes 
        inv_deg_e = hg.D_e_neg_1.values()
        inv_deg_e = torch.nan_to_num(inv_deg_e)
        edge_feat_norm = torch.einsum('ij,i->ij', edge_feat, inv_deg_e)

        node_feat = hg.e2v(edge_feat_norm, aggr = 'sum')
        
        return node_feat, edge_feat 

class HyperScatteringModule(nn.Module):
    def __init__(self, in_channels, trainable_laziness = False, trainable_scales = False, activation = "blis", fixed_weights=True):

        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = HyperDiffusion(in_channels, in_channels, trainable_laziness, fixed_weights)
        # self.diffusion_layer2 = Diffuse(
        #     4 * in_channels, 4 * in_channels, trainable_laziness
        # )
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [1, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], requires_grad=trainable_scales))

        if activation == "blis":
            self.activations = [lambda x: torch.relu(x), lambda x: torch.relu(-x)]
        elif activation == None:
            self.activations = [lambda x : x]
        elif activation == "modulus":
            self.activations = [lambda x: torch.abs(x)]

    def forward(self, X, hg):

        """ This performs  Px with P = 1/2(I + AD^-1) (column stochastic matrix) at the different scales"""

        #x, edge_index = data.x, data.edge_index
        features = X.shape[1]
        #s0 = X[:,:,None]
        node_features = [X]
        # i need to generate identity features (or null) on the edges
        Y0 = torch.zeros((hg.num_e, features))
        edge_features = [Y0]
        #import pdb; pdb.set_trace()
        for i in range(16):
            node_feat, edge_feat = self.diffusion_layer1(node_features[-1], hg, edge_features[-1])
            node_features.append(node_feat)
            edge_features.append(edge_feat)
        # for j in range(len(avgs)):
        #     # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
        #     avgs[j] = avgs[j][None, :, :, :]  
        # Combine the diffusion levels into a single tensor.
        import pdb; pdb.set_trace()
        diffusion_levels = torch.cat(node_features)
        edge_diffusion_levels = torch.cat(edge_features)
        
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter0 = avgs[0] - avgs[1]
        # filter1 = avgs[1] - avgs[2] 
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16] 
        # filter5 = avgs[16]

        # the shapes here are all wrong

        wavelet_coeffs = torch.einsum("ij,jklm->iklm", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        #subtracted = subtracted.view(6, x.shape[0], x.shape[1]) # reshape into given input shape
        activated = [self.activations[i](wavelet_coeffs) for i in range(len(self.activations))]
        
        s = torch.cat(activated, axis=-1).transpose(1,0)
        
        return s
    
    def out_features(self):
        return 12 * self.in_channels

        
if __name__ == "__main__":
    num_vertices = 15
    hg = dhg.random.uniform_hypergraph_Gnp(3,num_vertices, .4)
    signal_features = 2
    X = torch.rand(num_vertices, signal_features)
    blis_layer = HyperScatteringModule(signal_features)
    blis_layer(X, hg)