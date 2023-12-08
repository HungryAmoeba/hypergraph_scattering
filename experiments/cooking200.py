import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from dhg import Hypergraph
from dhg.data import Cooking200, CoauthorshipCora
from dhg.models import HGNN, HGNNP
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import argparse

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

# run training on the datasets provided in dhg, such as cooking200 and coauthorshipcora 









def train(net, X, A, lbls, train_idx, optimizer, epoch, device, args):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    if args.model == 'HSN':
        Y = torch.zeros(A.num_e, X.shape[1]).to(device)
        outs, edge_pred = net(A, X, Y)
    else:
        outs = net(X,A)
    outs, lbls = outs[train_idx], lbls[train_idx]

    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, args, device, test=False):
    net.eval()
    if args.model == 'HSN':
        Y = torch.zeros(A.num_e, X.shape[1]).to(device)
        outs, edge_pred = net(A, X, Y)
    else:
        outs = net(X,A)
    #outs = net(X,A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser Example')

    # Adding arguments
    parser.add_argument('--model', choices=['HSN', 'HGNN', 'HGNNP', 'HyperGCN', 'DHCF', 'HNHN', 'UniGCN', 'UniGAT', 'UniSAGE', 'UniGIN'],
                        help='Choose a model (HSN, HGNN, HGNNP, HyperGCN, DHCF, HNHN, UniGCN, UniGAT, UniSAGE, UniGIN)')
    parser.add_argument('--dataset', default='Cooking200', help='Specify the dataset (default: Cooking200)')

    # Parsing the arguments
    args = parser.parse_args()

    set_seed(42)
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    
    if args.dataset == 'Cooking200':
        data = Cooking200()
    if args.dataset == 'Cora':
        data = CoauthorshipCora()

    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    if args.model == 'HGNNP':
        net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=True)
    if args.model == 'HSN':
        net = HSN(X.shape[1], 32, data["num_classes"], activation = "modulus")
    #net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch, device, args)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask, args, device)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, args, device, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)


