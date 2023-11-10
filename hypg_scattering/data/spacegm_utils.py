# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:53:12 2021

@author: zhenq
"""
import numpy as np
import pickle

import numpy as np
import networkx as nx
from scipy.stats import rankdata
import warnings
import torch
import torch_geometric as tg


EDGE_TYPES = {
    "neighbor": 0,
    "distant": 1,
    "self": 2,
}

# Metadata for the example dataset
BIOMARKERS_UPMC = [
    "CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", "CD3e",
    "CD4", "CD45", "CD45RA", "CD45RO", "CD68", "CD8", "CollagenIV", "HLA-DR",
    "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA",
]

CELL_TYPE_MAPPING_UPMC = {
    'APC': 0,
    'B cell': 1,
    'CD4 T cell': 2,
    'CD8 T cell': 3,
    'Granulocyte': 4,
    'Lymph vessel': 5,
    'Macrophage': 6,
    'Naive immune cell': 7,
    'Stromal / Fibroblast': 8,
    'Tumor': 9,
    'Tumor (CD15+)': 10,
    'Tumor (CD20+)': 11,
    'Tumor (CD21+)': 12,
    'Tumor (Ki67+)': 13,
    'Tumor (Podo+)': 14,
    'Vessel': 15,
    'Unassigned': 16,
}

CELL_TYPE_FREQ_UPMC = {
    'APC': 0.038220815854819415,
    'B cell': 0.06635091324932002,
    'CD4 T cell': 0.09489001514723677,
    'CD8 T cell': 0.07824503590797544,
    'Granulocyte': 0.026886102677111563,
    'Lymph vessel': 0.006429085023448621,
    'Macrophage': 0.10251942892685563,
    'Naive immune cell': 0.033537398925429215,
    'Stromal / Fibroblast': 0.07692583870182068,
    'Tumor': 0.10921293560435145,
    'Tumor (CD15+)': 0.06106975782857908,
    'Tumor (CD20+)': 0.02098925720318548,
    'Tumor (CD21+)': 0.053892044158901406,
    'Tumor (Ki67+)': 0.13373768013421947,
    'Tumor (Podo+)': 0.06276108605978743,
    'Vessel': 0.034332604596958326,
    'Unassigned': 0.001,
}

def process_edge_distance(G,
                          edge_ind,
                          log_distance_lower_bound=2.,
                          log_distance_upper_bound=5.,
                          **kwargs):
    """ Process edge distance, distance will be log-transformed and min-max normalized

    Default parameters assume distances are usually within the range: 10-100 pixels / 3.7-37 um

    Args:
        G (nx.Graph): full cellular graph of the region
        edge_ind (int): target edge index
        log_distance_lower_bound (float): lower bound for log-transformed distance
        log_distance_upper_bound (float): upper bound for log-transformed distance

    Returns:
        list: list of normalized log-transformed distance
    """
    dist = G.edges[edge_ind]["distance"]
    log_dist = np.log(dist + 1e-5)
    _d = np.clip((log_dist - log_distance_lower_bound) /
                 (log_distance_upper_bound - log_distance_lower_bound), 0, 1)
    return [_d]

def process_neighbor_composition(G,
                                 node_ind,
                                 cell_type_mapping=None,
                                 neighborhood_size=10,
                                 **kwargs):
    """ Calculate the composition vector of k-nearest neighboring cells

    Args:
        G (nx.Graph): full cellular graph of the region
        node_ind (int): target node index
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        neighborhood_size (int): number of nearest neighbors to consider

    Returns:
        comp_vec (list): composition vector of k-nearest neighboring cells
    """
    center_coord = G.nodes[node_ind]["center_coord"]

    def node_dist(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2), ord=2)

    radius = 1
    neighbors = {}
    while len(neighbors) < 2 * neighborhood_size and radius < 5:
        radius += 1
        ego_g = nx.ego_graph(G, node_ind, radius=radius)
        neighbors = {n: feat_dict["center_coord"] for n, feat_dict in ego_g.nodes.data()}

    closest_neighbors = sorted(neighbors.keys(), key=lambda x: node_dist(center_coord, neighbors[x]))
    closest_neighbors = closest_neighbors[1:(neighborhood_size + 1)]

    comp_vec = np.zeros((len(cell_type_mapping),))
    for n in closest_neighbors:
        cell_type = cell_type_mapping[G.nodes[n]["cell_type"]]
        comp_vec[cell_type] += 1
    comp_vec = list(comp_vec / comp_vec.sum())
    return comp_vec

def process_biomarker_expression(G,
                                 node_ind,
                                 biomarkers=None,
                                 biomarker_expression_process_method='raw',
                                 biomarker_expression_lower_bound=-3,
                                 biomarker_expression_upper_bound=3,
                                 **kwargs):
    """ Process biomarker expression

    Args:
        G (nx.Graph): full cellular graph of the region
        node_ind (int): target node index
        biomarkers (list): list of biomarkers
        biomarker_expression_process_method (str): process method, one of 'raw', 'linear', 'log', 'rank'
        biomarker_expression_lower_bound (float): lower bound for min-max normalization, used for 'linear' and 'log'
        biomarker_expression_upper_bound (float): upper bound for min-max normalization, used for 'linear' and 'log'

    Returns:
        list: processed biomarker expression values
    """

    bm_exp_dict = G.nodes[node_ind]["biomarker_expression"]
    bm_exp_vec = []
    for bm in biomarkers:
        if bm_exp_dict is None or bm not in bm_exp_dict:
            bm_exp_vec.append(0.)
        else:
            bm_exp_vec.append(float(bm_exp_dict[bm]))

    bm_exp_vec = np.array(bm_exp_vec)
    lb = biomarker_expression_lower_bound
    ub = biomarker_expression_upper_bound

    if biomarker_expression_process_method == 'raw':
        return list(bm_exp_vec)
    elif biomarker_expression_process_method == 'linear':
        bm_exp_vec = np.clip(bm_exp_vec, lb, ub)
        bm_exp_vec = (bm_exp_vec - lb) / (ub - lb)
        return list(bm_exp_vec)
    elif biomarker_expression_process_method == 'log':
        bm_exp_vec = np.clip(np.log(bm_exp_vec + 1e-9), lb, ub)
        bm_exp_vec = (bm_exp_vec - lb) / (ub - lb)
        return list(bm_exp_vec)
    elif biomarker_expression_process_method == 'rank':
        bm_exp_vec = rankdata(bm_exp_vec, method='min')
        num_exp = len(bm_exp_vec)
        bm_exp_vec = (bm_exp_vec - 1) / (num_exp - 1)
        return list(bm_exp_vec)
    else:
        raise ValueError("expression process method %s not recognized" % biomarker_expression_process_method)

def process_feature(G, feature_item, node_ind=None, edge_ind=None, **feature_kwargs):
    """ Process a single node/edge feature item

    The following feature items are supported, note that some of them require
    keyword arguments in `feature_kwargs`:

    Node features:
        - feature_item: "cell_type"
            (required) "cell_type_mapping"
        - feature_item: "center_coord"
        - feature_item: "biomarker_expression"
            (required) "biomarkers",
            (optional) "biomarker_expression_process_method",
            (optional, if method is "linear" or "log") "biomarker_expression_lower_bound",
            (optional, if method is "linear" or "log") "biomarker_expression_upper_bound"
        - feature_item: "neighborhood_composition"
            (required) "cell_type_mapping",
            (optional) "neighborhood_size"
        - other additional feature items stored in the node attributes
            (see `graph_build.construct_graph_for_region`, argument `cell_features_file`)

    Edge features:
        - feature_item: "edge_type"
        - feature_item: "distance"
            (optional) "log_distance_lower_bound",
            (optional) "log_distance_upper_bound"

    Args:
        G (nx.Graph): full cellular graph of the region
        feature_item (str): feature item
        node_ind (int): target node index (if feature item is node feature)
        edge_ind (tuple): target edge index (if feature item is edge feature)
        feature_kwargs (dict): arguments for processing features

    Returns:
        v (list): feature vector
    """
    # Node features
    if node_ind is not None and edge_ind is None:
        if feature_item == "cell_type":
            # Integer index of the cell type
            assert "cell_type_mapping" in feature_kwargs, \
                "'cell_type_mapping' is required in the kwargs for feature item 'cell_type'"
            v = [feature_kwargs["cell_type_mapping"][G.nodes[node_ind]["cell_type"]]]
            return v
        elif feature_item == "center_coord":
            # Coordinates of the cell centroid
            v = list(G.nodes[node_ind]["center_coord"])
            return v
        elif feature_item == "biomarker_expression":
            # Biomarker expression of the cell
            assert "biomarkers" in feature_kwargs, \
                "'biomarkers' is required in the kwargs for feature item 'biomarker_expression'"
            v = process_biomarker_expression(G, node_ind, **feature_kwargs)
            return v
        elif feature_item == "neighborhood_composition":
            # Composition vector of the k-nearest neighboring cells
            assert "cell_type_mapping" in feature_kwargs, \
                "'cell_type_mapping' is required in the kwargs for feature item 'neighborhood_composition'"
            v = process_neighbor_composition(G, node_ind, **feature_kwargs)
            return v
        elif feature_item in G.nodes[node_ind]:
            # Additional features specified by users, e.g. "SIZE" in the example
            v = [G.nodes[node_ind][feature_item]]
            return v
        else:
            raise ValueError("Feature %s not found in the node attributes of graph %s, node %s" %
                             (feature_item, G.region_id, str(node_ind)))

    # Edge features
    elif edge_ind is not None and node_ind is None:
        if feature_item == "edge_type":
            v = [EDGE_TYPES[G.edges[edge_ind]["edge_type"]]]
            return v
        elif feature_item == "distance":
            v = process_edge_distance(G, edge_ind, **feature_kwargs)
            return v
        elif feature_item in G.edges[edge_ind]:
            v = [G.edges[edge_ind][feature_item]]
            return v
        else:
            raise ValueError("Feature %s not found in the edge attributes of graph %s, edge %s" %
                             (feature_item, G.region_id, str(edge_ind)))

    else:
        raise ValueError("One of node_ind or edge_ind should be specified")


def get_cell_type_metadata(nx_graph_files):
    """Find all unique cell types from a list of cellular graphs

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        cell_type_freq (dict): mapping of unique cell types to their frequency
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]
    cell_type_mapping = {}
    for g_f in nx_graph_files:
        G = pickle.load(open(g_f, 'rb'))
        assert 'cell_type' in G.nodes[0]
        for n in G.nodes:
            ct = G.nodes[n]['cell_type']
            if ct not in cell_type_mapping:
                cell_type_mapping[ct] = 0
            cell_type_mapping[ct] += 1
    unique_cell_types = sorted(cell_type_mapping.keys())
    unique_cell_types_ct = [cell_type_mapping[ct] for ct in unique_cell_types]
    unique_cell_type_freq = [count / sum(unique_cell_types_ct) for count in unique_cell_types_ct]
    cell_type_mapping = {ct: i for i, ct in enumerate(unique_cell_types)}
    cell_type_freq = dict(zip(unique_cell_types, unique_cell_type_freq))
    return cell_type_mapping, cell_type_freq


def get_biomarker_metadata(nx_graph_files):
    """Load all biomarkers from a list of cellular graphs

    Args:
        nx_graph_files (list/str): path/list of paths to cellular graph files (gpickle)

    Returns:
        shared_bms (list): list of biomarkers shared by all cells (intersect)
        all_bms (list): list of all biomarkers (union)
    """
    if isinstance(nx_graph_files, str):
        nx_graph_files = [nx_graph_files]
    all_bms = set()
    shared_bms = None
    for g_f in nx_graph_files:
        G = pickle.load(open(g_f, 'rb'))
        for n in G.nodes:
            bms = sorted(G.nodes[n]["biomarker_expression"].keys())
            for bm in bms:
                all_bms.add(bm)
            valid_bms = [
                bm for bm in bms if G.nodes[n]["biomarker_expression"][bm] == G.nodes[n]["biomarker_expression"][bm]]
            shared_bms = set(valid_bms) if shared_bms is None else shared_bms & set(valid_bms)
    shared_bms = sorted(shared_bms)
    all_bms = sorted(all_bms)
    return shared_bms, all_bms


def get_graph_splits(dataset,
                     split='random',
                     cv_k=5,
                     seed=None,
                     fold_mapping=None):
    """ Define train/valid split

    Args:
        dataset (CellularGraphDataset): dataset to split
        split (str): split method, one of 'random', 'fold'
        cv_k (int): number of splits for random split
        seed (int): random seed
        fold_mapping (dict): mapping of region ids to folds,
            fold could be coverslip, patient, etc.

    Returns:
        split_inds (list): fold indices for each region in the dataset
    """
    splits = {}
    region_ids = set([dataset.get_full(i).region_id for i in range(dataset.N)])
    _region_ids = sorted(region_ids)
    if split == 'random':
        if seed is not None:
            np.random.seed(seed)
        if fold_mapping is None:
            fold_mapping = {region_id: region_id for region_id in _region_ids}
        # `_ids` could be sample ids / patient ids / certain properties
        _folds = sorted(set(list(fold_mapping.values())))
        np.random.shuffle(_folds)
        cv_shard_size = len(_folds) / cv_k
        for i, region_id in enumerate(_region_ids):
            splits[region_id] = _folds.index(fold_mapping[region_id]) // cv_shard_size
    elif split == 'fold':
        # Split into folds, one fold per group
        assert fold_mapping is not None
        _folds = sorted(set(list(fold_mapping.values())))
        for i, region_id in enumerate(_region_ids):
            splits[region_id] = _folds.index(fold_mapping[region_id])
    else:
        raise ValueError("split mode not recognized")

    split_inds = []
    for i in range(dataset.N):
        split_inds.append(splits[dataset.get_full(i).region_id])
    return split_inds

def nx_to_tg_graph(G,
                   node_features=["cell_type",
                                  "biomarker_expression",
                                  "neighborhood_composition",
                                  "center_coord"],
                   edge_features=["edge_type",
                                  "distance"],
                   **feature_kwargs):
    """ Build pyg data objects from nx graphs

    Args:
        G (nx.Graph): full cellular graph of the region
        node_features (list, optional): list of node feature items
        edge_features (list, optional): list of edge feature items
        feature_kwargs (dict): arguments for processing features

    Returns:
        data_list (list): list of pyg data objects
    """
    data_list = []

    # Each connected component of the cellular graph will be a separate pyg data object
    # Usually there should only be one connected component for each cellular graph
    for inds in nx.connected_components(G):
        # Skip small connected components
        if len(inds) < len(G) * 0.1:
            continue
        sub_G = G.subgraph(inds)

        # Relabel nodes to be consecutive integers, note that node indices are
        # not meaningful here, cells are identified by the key "cell_id" in each node
        mapping = {n: i for i, n in enumerate(sorted(sub_G.nodes))}
        sub_G = nx.relabel.relabel_nodes(sub_G, mapping)
        assert np.all(sub_G.nodes == np.arange(len(sub_G)))

        # Append node and edge features to the pyg data object
        data = {"x": [], "edge_attr": [], "edge_index": []}
        for node_ind in sub_G.nodes:
            feat_val = []
            for key in node_features:
                feat_val.extend(process_feature(sub_G, key, node_ind=node_ind, **feature_kwargs))
            data["x"].append(feat_val)

        for edge_ind in sub_G.edges:
            feat_val = []
            for key in edge_features:
                feat_val.extend(process_feature(sub_G, key, edge_ind=edge_ind, **feature_kwargs))
            data["edge_attr"].append(feat_val)
            data["edge_index"].append(edge_ind)
            data["edge_attr"].append(feat_val)
            data["edge_index"].append(tuple(reversed(edge_ind)))

        for key, item in data.items():
            data[key] = torch.tensor(item)
        data['edge_index'] = data['edge_index'].t().long()
        data = tg.data.Data.from_dict(data)
        data.num_nodes = sub_G.number_of_nodes()
        data.region_id = G.region_id
        data_list.append(data)
    return data_list


def get_feature_names(features, cell_type_mapping=None, biomarkers=None):
    """ Helper fn for getting a list of feature names from a list of feature items

    Args:
        features (list): list of feature items
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        biomarkers (list): list of biomarkers

    Returns:
        feat_names(list): list of feature names
    """
    feat_names = []
    for feat in features:
        if feat in ["distance", "cell_type", "edge_type"]:
            # feature "cell_type", "edge_type" will be a single integer indice
            # feature "distance" will be a single float value
            feat_names.append(feat)
        elif feat == "center_coord":
            # feature "center_coord" will be a tuple of two float values
            feat_names.extend(["center_coord-x", "center_coord-y"])
        elif feat == "biomarker_expression":
            # feature "biomarker_expression" will contain a list of biomarker expression values
            feat_names.extend(["biomarker_expression-%s" % bm for bm in biomarkers])
        elif feat == "neighborhood_composition":
            # feature "neighborhood_composition" will contain a composition vector of the immediate neighbors
            # The vector will have the same length as the number of unique cell types
            feat_names.extend(["neighborhood_composition-%s" % ct
                               for ct in sorted(cell_type_mapping.keys(), key=lambda x: cell_type_mapping[x])])
        else:
            warnings.warn("Using additional feature: %s" % feat)
            feat_names.append(feat)
    return feat_names