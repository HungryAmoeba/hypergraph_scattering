import unittest
import torch
import sys
from data import get_HGDataset
from torch_geometric.data import Data

class TestGetHGDataset(unittest.TestCase):
    def test_empty_dataset(self):
        original_dataset = []
        hgdataset = get_HGDataset(original_dataset)
        self.assertEqual(len(hgdataset), 0)

    def test_single_graph(self):

        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        num_nodes = 3
        node_features = torch.tensor([[1, 2], [3, 4], [5, 6]])
        labels = torch.tensor([0, 1, 0])
        graph_dat = Data(edge_index=edge_index, num_nodes=num_nodes, x=node_features, y=labels)
        original_dataset = [graph_dat]

        # Call the function
        hgdataset = get_HGDataset(original_dataset)

        # Check the result
        self.assertEqual(len(hgdataset), 1)
        self.assertEqual(hgdataset[0].num_nodes, num_nodes)
        self.assertEqual(hgdataset[0].num_edges, 3)
        self.assertTrue(torch.all(torch.eq(hgdataset[0].x, node_features)))
        self.assertTrue(torch.all(torch.eq(hgdataset[0].x, node_features)))
        self.assertTrue(torch.all(torch.eq(hgdataset[0].y, labels)))

    def test_multiple_graphs(self):

        # Create multiple graph dataset
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

        # Call the function
        hgdataset = get_HGDataset(original_dataset)

        # Check the result
        self.assertEqual(len(hgdataset), 2)
        self.assertEqual(hgdataset[0].num_nodes, num_nodes_1)
        self.assertEqual(hgdataset[0].num_edges, 3)
        self.assertTrue(torch.all(torch.eq(hgdataset[0].x, node_features_1)))
        self.assertTrue(torch.all(torch.eq(hgdataset[0].y, labels_1)))

        self.assertEqual(hgdataset[1].num_nodes, num_nodes_2)
        self.assertEqual(hgdataset[1].num_edges, 1)
        self.assertTrue(torch.all(torch.eq(hgdataset[1].x, node_features_2)))
        self.assertTrue(torch.all(torch.eq(hgdataset[1].y, labels_2)))

if __name__ == '__main__':
    unittest.main()