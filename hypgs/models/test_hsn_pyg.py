#from hypg_scattering.models.hyper_scattering_net import HSN
import sys
# fix this later!
# sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
sys.path.append('../../')
from hypgs.models.hyper_scattering_net import HyperDiffusion
from hypgs.models.hsn_pyg import HGDiffsion
from hypgs.utils.data import get_HyperGraphData
import dhg 
import torch
import unittest


def gen_hypg_deg_dataset(num_v, num_e, num_graphs):
    hypergraph_dataset = [dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first') for _ in range(num_graphs)]
    return hypergraph_dataset

class TestHGDiffsion(unittest.TestCase):
    def test_vs_HyperDiffusion(self, trainable_laziness = False, fixed_weights=True, test=False):
        if not test:
            return
        num_graphs = 100
        num_v = 10
        num_e = 20 
        # generate graph dataset
        train_dataset = gen_hypg_deg_dataset(num_v, num_e, num_graphs)
        hg = train_dataset[0]
        starting_features = torch.randn(num_v, num_v)
        signal_features = num_v
        num_e_hg = hg.num_e 
        Y = torch.randn(num_e_hg, num_v)
        hgdata = get_HyperGraphData(hg, starting_features, Y, None)
        hyd = HyperDiffusion(in_channels=signal_features, out_channels=signal_features, trainable_laziness=trainable_laziness, fixed_weights=fixed_weights)
        hyd_pyg = HGDiffsion(in_channels=signal_features, out_channels=signal_features, trainable_laziness=trainable_laziness, fixed_weights=fixed_weights, normalize='right')
        if trainable_laziness:
            hyd_pyg.lazy_layer.weights = hyd.lazy_layer.weights
        if not fixed_weights:
            hyd_pyg.lin_neigh = hyd.lin_neigh
            hyd_pyg.lin_self = hyd.lin_self
        node_feat, edge_feat = hyd(hg, starting_features, Y)
        node_feat2, edge_feat2 = hyd_pyg(x=hgdata.x, hyperedge_index=hgdata.edge_index, hyperedge_attr=hgdata.edge_attr)
        assert torch.isclose(node_feat2, node_feat).all(), "Node features are not equal"
        assert torch.isclose(edge_feat2, edge_feat).all(), "Edge features are not equal"
    
    def test_all(self):
        self.test_vs_HyperDiffusion(trainable_laziness = False, fixed_weights=True, test=True)
        self.test_vs_HyperDiffusion(trainable_laziness = True, fixed_weights=True, test=True)
        self.test_vs_HyperDiffusion(trainable_laziness = False, fixed_weights=False, test=True)
        self.test_vs_HyperDiffusion(trainable_laziness = True, fixed_weights=False, test=True)

if __name__ == '__main__':
    unittest.main()