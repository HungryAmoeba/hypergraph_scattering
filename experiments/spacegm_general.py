# original author: EDB

import torch
import os
import sys
import argparse 

import itertools

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
#from hypgg.ScatteringTransforms.pyg_scattering_transform import GraphScatteringTransform
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pdb
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import yaml
from hypgs import DATA_DIR
from hypgs.utils.hypergraph_utils import CliqueHyperEdgeTransform


# fix this later!
# sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
from hypgs.data.spacegm import CellularGraphDataset
from hypgs.data.spacegm_transforms import AddCenterCellType, AddGraphLabel, FeatureMask 

parser = argparse.ArgumentParser(description='Process scattering on a dataset')

# Load the YAML file
# with open('../hypg_scattering/utils/config.yaml', 'r') as file:
#     config = yaml.safe_load(file)


print(sys.argv)
n = len(sys.argv)
if n != 5:
    print('Error. Usage: python spacegm_general.py dataset_name delete_processed classifier_name scattering_type')
    print("example: python spacegm_general.py upmc 1 svm asym")
    exit()

dataset_name = str(sys.argv[1])

delete = bool(int(sys.argv[2]))
classifier = str(sys.argv[3])

if classifier not in {'svm', 'logistic'}:
    print('supported classifiers: svm and logistic')
    exit()

print(f'Delete existing transform = {delete}')

root_dir = os.path.join(DATA_DIR,"upmc")

WAVELET_TYPE = 2
MAX_SCALE = 5
NUM_LAYERS = 3
HIGHEST_MOMENT = 1
AGGREGATION_TYPE = 'L1'

scattering_type = str(sys.argv[4])
print(scattering_type)
if scattering_type not in {'blis' , 'asym'}:
    print('supported scattering types: blip and asym')
    exit()

#scatterTransform = GraphScatteringTransform(scattering_type, WAVELET_TYPE, MAX_SCALE, NUM_LAYERS, HIGHEST_MOMENT)
#scatterTransform = GraphScatteringTransform(scattering_type, WAVELET_TYPE, MAX_SCALE, NUM_LAYERS, AGGREGATION_TYPE)
#transformations = T.Compose([scatterTransform])
#transformations = None

# perhaps it is necessary to delete pre_transform every time it is used?
processed_data_dir = os.path.join(root_dir, f'tg_graph')
if delete:
    try:
        shutil.rmtree(processed_data_dir)
        print(f"Directory '{processed_data_dir}' successfully deleted.")
    except FileNotFoundError:
        print(f"Directory '{processed_data_dir}' does not exist.")
    except PermissionError:
        print(f"Permission denied. Unable to delete directory '{processed_data_dir}'.")
    except Exception as e:
        print(f"An error occurred while deleting directory '{processed_data_dir}': {str(e)}")


dataset_root = root_dir
dataset_kwargs = {
    'transform': [],
    'pre_transform': [CliqueHyperEdgeTransform()], # removed the scattering transform atm
    'pre_pre_transform': None,
    'raw_folder_name': 'graphs',  # os.path.join(dataset_root, "graph") is the folder where we saved nx graphs
    'processed_folder_name': 'tg_graph',  # processed dataset files will be stored here
    'preprocessed_folder_name': 'tg_pre_graph',
    #'node_features': ["cell_type", "SIZE", "biomarker_expression", "neighborhood_composition", "center_coord"],  # There are all the cellular features that we want the dataset to compute
    'node_features': ["cell_type", "SIZE", "biomarker_expression", "center_coord"],  # There are all the cellular features that we want the dataset to compute
    'edge_features': ["edge_type", "distance"],  # edge (cell pair) features
    # this is usualy 3, but I using 0 to indicidate that the whole graph should be used
    'subgraph_size': 0,  # indicating we want to sample 3-hop subgraphs from these regions (for training/inference), this is a core parameter for SPACE-GM.
    'subgraph_source': 'on-the-fly',
    'subgraph_allow_distant_edge': True,
    'subgraph_radius_limit': 200.,
}

feature_kwargs = {
    "biomarker_expression_process_method": "linear",
    "biomarker_expression_lower_bound": 0,
    "biomarker_expression_upper_bound": 18,
    "neighborhood_size": 10,
}
dataset_kwargs.update(feature_kwargs)

dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)


graph_label_file = os.path.join(dataset_root, "upmc_labels_renamed.csv")
transformers = [
    # `AddCenterCellType` will add `node_y` attribute to the subgraph for node-level prediction task
    # In this task we will mask the cell type of the center cell and use its neighborhood to predict the true cell type
    #AddCenterCellType(dataset),
    
    # `AddGraphLabel` will add `graph_y` and `graph_w` attributes to the subgraph for graph-level prediction task
    AddGraphLabel(graph_label_file, tasks=['survival_status']),
    
    # Transformer `FeatureMask` will zero mask all feature items not included in its argument
    # In this tutorial we perform training/inference using cell types and center cell's size feature
    #FeatureMask(dataset, use_center_node_features=['cell_type', 'SIZE'], use_neighbor_node_features=['cell_type']),
]
dataset.set_transforms(transformers)

breakpoint()

print(f'finished processing {dataset_name}')
import pdb; pdb.set_trace()
print("modeling using logistic regression")
n_splits = 5
print(f"training with {n_splits}-fold")

if classifier == 'logistic':
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 1000))
    print('initialized logistic regression')
else:
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    print('initialized SVM with RBF kernel')
kf = KFold(n_splits = n_splits)
loader = DataLoader(dataset, batch_size = 1, shuffle = True)

# Perform k-fold cross-validation
fold_scores = []
fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []
#import pdb; pdb.set_trace()
#dataset.x = torch.nan_to_num(dataset.x)
for train_indices, test_indices in kf.split(dataset):
    # Split the dataset into train and test folds
    train_data = [dataset[train_idx] for train_idx in train_indices]
    test_data = [dataset[test_idx] for test_idx in test_indices]

    train_dat_x = torch.cat([d.x for d in train_data])
    train_dat_y = torch.cat([d.graph_y for d in train_data])[:,0]

    test_dat_x = torch.cat([d.x for d in test_data])
    test_dat_y = torch.cat([d.graph_y for d in test_data])[:,0]

    # Train the logistic regression model
    model.fit(train_dat_x, train_dat_y)

    # Make predictions on the test fold
    y_pred = model.predict(test_dat_x)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_dat_y, y_pred)
    precision = precision_score(test_dat_y, y_pred, average='macro')
    recall = recall_score(test_dat_y, y_pred, average='macro')
    f1 = f1_score(test_dat_y, y_pred, average='macro')

    # Append scores and metrics to lists
    fold_scores.append(model.score(test_dat_x, test_dat_y))
    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)

# Calculate mean and standard deviation of scores
mean_score = sum(fold_scores) / n_splits
std_score = torch.tensor(fold_scores).std().item()

# Calculate mean and standard deviation of metrics
mean_accuracy = sum(fold_accuracies) / n_splits
std_accuracy = torch.tensor(fold_accuracies).std().item()
mean_precision = sum(fold_precisions) / n_splits
std_precision = torch.tensor(fold_precisions).std().item()
mean_recall = sum(fold_recalls) / n_splits
std_recall = torch.tensor(fold_recalls).std().item()
mean_f1_score = sum(fold_f1_scores) / n_splits
std_f1_score = torch.tensor(fold_f1_scores).std().item()

# Print scores and performance metrics
print(f"Cross-Validation Scores: {fold_scores}")
print(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
print("")

print(f"Accuracy Scores: {fold_accuracies}")
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print("")

print(f"Precision Scores: {fold_precisions}")
print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print("")

print(f"Recall Scores: {fold_recalls}")
print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print("")

print(f"F1 Scores: {fold_f1_scores}")
print(f"Mean F1 Score: {mean_f1_score:.4f} ± {std_f1_score:.4f}")
