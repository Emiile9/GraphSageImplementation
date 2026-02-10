import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from Algo_Mini_Batch import AlgoMiniBatch
from dataset import GraphSageDataset
from train import train
from layers import MeanAggregator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

G = nx.barabasi_albert_graph(n=200, m=3)

feat_dim = 32

for node in G.nodes():
    G.nodes[node]["features"] = torch.randn(feat_dim)

hidden_dim = feat_dim
depth = 2

W = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(depth)])
agg = [MeanAggregator() for _ in range(depth)]
sample_size = [5 for _ in range(depth)]

model = AlgoMiniBatch(depth, W, F.relu, agg)

print("Nb paramètres :", sum(p.numel() for p in model.parameters()))

train(model, G, device, sampling_size=sample_size, epochs=20, learning_rate=3e-4, batch_size=64)