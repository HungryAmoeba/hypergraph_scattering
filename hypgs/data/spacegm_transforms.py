import numpy as np
import pandas as pd
from copy import deepcopy
import torch


class FeatureMask(object):
    """ Transformer object for masking features """
    def __init__(self,
                 dataset,
                 use_neighbor_node_features=None,
                 use_center_node_features=None,
                 use_edge_features=None,
                 **kwargs):
        """ Construct the transformer

        Args:
            dataset (CellularGraphDataset): dataset object
            use_neighbor_node_features (list): list of node feature items to use
                for non-center nodes, all other features will be masked out
            use_center_node_features (list): list of node feature items to use
                for the center node, all other features will be masked out
            use_edge_features (list): list of edge feature items to use,
                all other features will be masked out
        """

        self.node_feature_names = dataset.node_feature_names
        self.edge_feature_names = dataset.edge_feature_names

        self.use_neighbor_node_features = use_neighbor_node_features if \
            use_neighbor_node_features is not None else dataset.node_features
        self.use_center_node_features = use_center_node_features if \
            use_center_node_features is not None else dataset.node_features
        self.use_edge_features = use_edge_features if \
            use_edge_features is not None else dataset.edge_features

        self.center_node_feature_masks = [
            1 if any(name.startswith(feat) for feat in self.use_center_node_features)
            else 0 for name in self.node_feature_names]
        self.neighbor_node_feature_masks = [
            1 if any(name.startswith(feat) for feat in self.use_neighbor_node_features)
            else 0 for name in self.node_feature_names]

        self.center_node_feature_masks = \
            torch.from_numpy(np.array(self.center_node_feature_masks).reshape((-1,))).float()
        self.neighbor_node_feature_masks = \
            torch.from_numpy(np.array(self.neighbor_node_feature_masks).reshape((1, -1))).float()

    def __call__(self, data):
        data = deepcopy(data)
        if "center_node_index" in data:
            center_node_feat = data.x_og[data.center_node_index].detach().data.clone()
        else:
            center_node_feat = None
        data = self.transform_neighbor_node(data)
        data = self.transform_center_node(data, center_node_feat)
        return data

    def transform_neighbor_node(self, data):
        """Apply neighbor node feature masking"""
        data.x_og = data.x_og * self.neighbor_node_feature_masks
        return data

    def transform_center_node(self, data, center_node_feat=None):
        """Apply center node feature masking"""
        if center_node_feat is None:
            return data
        assert "center_node_index" in data
        center_node_feat = center_node_feat * self.center_node_feature_masks
        data.x_og[data.center_node_index] = center_node_feat
        return data


class AddCenterCellType(object):
    """Transformer for center cell type prediction"""
    def __init__(self, dataset, **kwargs):
        self.node_feature_names = dataset.node_feature_names
        self.cell_type_feat = self.node_feature_names.index('cell_type')
        # Assign a placeholder cell type for the center node
        self.placeholder_cell_type = max(dataset.cell_type_mapping.values()) + 1

    def __call__(self, data):
        assert "center_node_index" in data, \
            "Only subgraphs with center nodes are supported, cannot find `center_node_index`"
        #import pdb; pdb.set_trace()
        center_node_feat = data.x[data.center_node_index].detach().clone()
        center_cell_type = center_node_feat[self.cell_type_feat]
        data.node_y = center_cell_type.long().view((1,))
        return data


class AddCenterCellBiomarkerExpression(object):
    """Transformer for center cell biomarker expression prediction"""
    def __init__(self, dataset, **kwargs):
        self.node_feature_names = dataset.node_feature_names
        self.bm_exp_feat = np.array([
            i for i, feat in enumerate(self.node_feature_names)
            if feat.startswith('biomarker_expression')])

    def __call__(self, data):
        assert "center_node_index" in data, \
            "Only subgraphs with center nodes are supported, cannot find `center_node_index`"
        center_node_feat = data.x[data.center_node_index].detach().clone()
        center_cell_exp = center_node_feat[self.bm_exp_feat].float()
        data.node_y = center_cell_exp.view(1, -1)
        return data


class AddCenterCellIdentifier(object):
    """Transformer for adding another feature column for identifying center cell
    Helpful when predicting node-level tasks.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data):
        import pdb; pdb.set_trace()
        assert "center_node_index" in data, \
            "Only subgraphs with center nodes are supported, cannot find `center_node_index`"
        center_cell_identifier_column = torch.zeros((data.x_og.shape[0], 1), dtype=data.x_og.dtype)
        center_cell_identifier_column[data.center_node_index, 0] = 1.
        data.x = torch.cat([data.x, center_cell_identifier_column], dim=1)
        return data


class AddGraphLabel(object):
    """Transformer for adding graph-level task labels"""
    def __init__(self, graph_label_file, tasks=[], **kwargs):
        """ Construct the transformer

        Args:
            graph_label_file (str): path to the csv file containing graph-level
                task labels. This file should always have the first column as region id.
            tasks (list): list of tasks to use, corresponding to column names
                of the csv file. If empty, use all tasks in the file
        """
        self.label_df = pd.read_csv(graph_label_file, index_col=0)
        self.label_df.index = self.label_df.index.map(str)  # Convert index to str
        graph_tasks = list(self.label_df.columns) if len(tasks) == 0 else tasks
        self.tasks, self.class_label_weights = self.build_class_weights(graph_tasks)

    def build_class_weights(self, graph_tasks):
        valid_tasks = []
        class_label_weights = {}
        for task in graph_tasks:
            ar = list(self.label_df[task])
            valid_vals = [_y for _y in ar if _y == _y]
            unique_vals = set(valid_vals)
            if not all(v.__class__ in [int, float] for v in unique_vals):
                # Skip tasks with non-numeric labels
                continue
            valid_tasks.append(task)
            if len(unique_vals) > 5:
                # More than 5 unique values in labels, likely a regression task
                class_label_weights[task] = {_y: 1 for _y in unique_vals}
            else:
                # Classification task, compute class weights
                val_counts = {_y: valid_vals.count(_y) for _y in unique_vals}
                max_count = max(val_counts.values())
                class_label_weights[task] = {_y: max_count / val_counts[_y] for _y in unique_vals}
        return valid_tasks, class_label_weights

    def fetch_label(self, region_id, task_name):
        y = self.label_df[task_name].loc[region_id]
        if y != y:
            y = 0
            w = 0
        else:
            w = self.class_label_weights[task_name][y]
        return y, w

    def __call__(self, data):
        graph_y = []
        graph_w = []
        for task in self.tasks:
            y, w = self.fetch_label(data.region_id, task)
            graph_y.append(y)
            graph_w.append(w)
        data.graph_y = torch.from_numpy(np.array(graph_y).reshape((1, -1)))
        data.graph_w = torch.from_numpy(np.array(graph_w).reshape((1, -1)))
        return data
