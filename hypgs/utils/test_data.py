import unittest
# from data import get_HGDataset
from torch_geometric.data import Data
from torch_geometric.data.hypergraph_data import HyperGraphData
from dhg import Hypergraph
import torch
from data import HGDataset

class TestHGDataset(unittest.TestCase):
    def test_len(self):
        # Create a dataset with 3 data items
        edge_index_1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        num_nodes_1 = 3
        node_features_1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
        labels_1 = torch.tensor([0, 1, 0])
        graph_dat_1 = Data(edge_index=edge_index_1, num_nodes=num_nodes_1, x=node_features_1, y=labels_1)
        edge_index_2 = torch.tensor([[0, 1], [1, 0]])
        num_nodes_2 = 2
        node_features_2 = torch.tensor([[7, 8], [9, 10]])
        labels_2 = torch.tensor([1, 0])
        graph_dat_2 = Data(edge_index=edge_index_2, num_nodes=num_nodes_2, x=node_features_2, y=labels_2)
        edge_index_3 = torch.tensor([[0, 1], [1, 0]])
        num_nodes_3 = 2
        node_features_3 = torch.tensor([[11, 12], [13, 14]])
        labels_3 = torch.tensor([0, 1])
        graph_dat_3 = Data(edge_index=edge_index_3, num_nodes=num_nodes_3, x=node_features_3, y=labels_3)

        original_dataset = [graph_dat_1, graph_dat_2, graph_dat_3]

        dataset = HGDataset(original_dataset)
        self.assertEqual(len(dataset), 3)

    def test_get(self):
        # Create a dataset with 2 data items
        edge_index_1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        num_nodes_1 = 3
        node_features_1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
        labels_1 = torch.tensor([0, 1, 0])
        graph_dat_1 = Data(edge_index=edge_index_1, num_nodes=num_nodes_1, x=node_features_1, y=labels_1)
        edge_index_2 = torch.tensor([[0, 1], [1, 0]])
        num_nodes_2 = 2
        node_features_2 = torch.tensor([[7, 8], [9, 10]])
        labels_2 = torch.tensor([1, 0])
        graph_dat_2 = Data(edge_index=edge_index_2, num_nodes=num_nodes_2, x=node_features_2, y=labels_2)

        original_dataset = [graph_dat_1, graph_dat_2]

        dataset = HGDataset(original_dataset)

        # Get the first data item
        data = dataset.get(0)

        # Check if the returned data item is of type Data
        self.assertIsInstance(data, Data)

    def test_to_hg_func_default(self):
        # Create a dataset with 1 data item
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        num_nodes = 3
        node_features = torch.tensor([[1, 2], [3, 4], [5, 6]])
        labels = torch.tensor([0, 1, 0])
        graph_dat = Data(edge_index=edge_index, num_nodes=num_nodes, x=node_features, y=labels)
        original_dataset = [graph_dat]

        # Create a dataset using the default to_hg_func
        dataset = HGDataset(original_dataset)

        # Get the first data item
        data = dataset.get(0)

        # Check if the data item has been converted to a hypergraph
        self.assertIsInstance(data, HyperGraphData)

    def test_to_hg_func_custom(self):
        # Define a custom to_hg_func
        custom_to_hg_func = lambda g: Hypergraph.from_graph_kHop(g, k=2)

        # Create a dataset with 1 data item
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        num_nodes = 3
        node_features = torch.tensor([[1, 2], [3, 4], [5, 6]])
        labels = torch.tensor([0, 1, 0])
        graph_dat = Data(edge_index=edge_index, num_nodes=num_nodes, x=node_features, y=labels)
        original_dataset = [graph_dat]

        # Create a dataset using the custom to_hg_func
        dataset = HGDataset(original_dataset, to_hg_func=custom_to_hg_func)

        # Get the first data item
        data = dataset.get(0)

        # Check if the data item has been converted to a hypergraph using the custom logic
        self.assertIsInstance(data, HyperGraphData)

if __name__ == '__main__':
    unittest.main()