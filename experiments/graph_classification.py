from torch_geometric.datasets import TUDataset
import sys
sys.path.append('..')
from dhg import Hypergraph
from hypgs.models.hsn_pyg import HSN
from hypgs.utils.data import HGDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import random_split
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from sklearn.model_selection import KFold
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name')


datasets = ['MUTAG', 'ENZYMES', 'PROTEINS']

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_name = args.dataset
    assert dataset_name in datasets, f"Dataset name must be one of {datasets}"
    original_dataset = TUDataset(root='../data/', name=dataset_name, use_node_attr=True)
    to_hg_func = lambda g: Hypergraph.from_graph_kHop(g, 1)
    dataset = HGDataset(original_dataset, to_hg_func)
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    # Setup KFold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Prepare to collect metrics
    all_test_metrics = []

    for fold, (train_ids, test_ids) in enumerate(kf.split(dataset)):
        # Split dataset into train and test for the current fold
        train_subset = [dataset[i] for i in train_ids]
        test_subset = [dataset[i] for i in test_ids]

        # Further split train_subset into training and validation sets
        train_size = int(0.85 * len(train_subset))
        train_data, val_data = random_split(train_subset, [train_size, len(train_subset) - train_size])

        # Create data loaders for train, validation, and test
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        # Initialize model and trainer for the current fold
        model = HSN(in_channels=in_channels, 
                    hidden_channels=64,
                    out_channels=out_channels, 
                    trainable_laziness=False,
                    trainable_scales=False, 
                    activation='modulus', 
                    fixed_weights=True, 
                    layout=['hsm', 'hsm'], 
                    normalize='right', 
                    pooling='max',
                    task='classification')
        
        # Early stopping and logger
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
        logger = TensorBoardLogger("lightning_logs", name=f"my_model_fold_{fold}")

        trainer = pl.Trainer(
            max_epochs=100,
            logger=logger,
            callbacks=[early_stopping_callback]
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Evaluate the model on the test set and collect metrics
        test_result = trainer.test(model, test_loader)[0]
        test_result['fold'] = fold
        all_test_metrics.append(test_result)

    # Export test metrics to CSV
    df_metrics = pd.DataFrame(all_test_metrics)
    df_metrics.to_csv(f"{dataset_name}_test_metrics.csv", index=False)

# # Assuming 'dataset' is your PyTorch Dataset
# dataset_size = len(dataset)
# train_size = int(0.7 * dataset_size)
# val_size = int(0.15 * dataset_size)
# test_size = dataset_size - (train_size + val_size)

# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model = HSN(in_channels=7, 
#               hidden_channels=64,
#               out_channels = 2, 
#               trainable_laziness = False,
#               trainable_scales = False, 
#               activation = 'modulus', 
#               fixed_weights=True, 
#               layout=['hsm', 'hsm'], 
#               normalize='right', 
#               pooling='max',
#               task='classification',
#         )

# # Early stopping callback based on validation loss
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
# logger = TensorBoardLogger("lightning_logs", name="my_model")

# trainer = pl.Trainer(
#     max_epochs=100,  # Adjust as necessary
#     logger=logger,
#     callbacks=[early_stopping_callback],
# )

# # Assuming you have defined train_loader and val_loader
# trainer.fit(model, train_loader, val_loader)  # Corrected argument names
# # Test the model
# trainer.test(model, test_loader)