"""Fichier pour tester les fonctions utils sur petites instances"""

import networkx as nx
import numpy as np
import random
import torch
from src.utils.utils import get_n_neighbours
from src.layers import MeanAggregator, MaxPoolingAggregator, LSTMAggregator

G = nx.Graph()
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 4), (3, 5)]
G.add_edges_from(edges)

features = np.array(
    [
        [1.0, 0.1, 5.0],
        [0.2, 0.8, 1.0],
        [1.5, 0.3, 4.0],
        [0.1, 0.9, 0.0],
        [1.2, 0.2, 4.5],
        [0.0, 0.7, 0.5],
    ],
    dtype=np.float32,
)

# Optionnel : On peut attacher les features aux nœuds dans NetworkX
for i in range(len(features)):
    G.nodes[i]["features"] = features[i]

MeanAggregator_instance = MeanAggregator()
MaxPoolingAggregator_instance = MaxPoolingAggregator(in_features=3, out_features=4)
LSTMAggregator_instance = LSTMAggregator(in_features=3, out_features=4)

node_0_neighbors = get_n_neighbours(G, 0, 2)
node_1_neighbors = get_n_neighbours(G, 1, 2)


batch_neighbors_feats = np.array(
    [features[node_0_neighbors], features[node_1_neighbors]]
)

input_batch = torch.FloatTensor(batch_neighbors_feats)
# Forme : (2, 2, 3)
with torch.no_grad():
    output_batch_mean = MeanAggregator_instance.aggregate(input_batch)
    output_batch_max_pool = MaxPoolingAggregator_instance.aggregate(input_batch)
    output_batch_lstm = LSTMAggregator_instance.aggregate(input_batch)
# Forme : (2, 3) -> Une moyenne par nœud du batch
print("Input batch :", input_batch)
print("Output with mean agg:", output_batch_mean)
print("Output with Max Pool Agg:", output_batch_max_pool)
print("Output with LSTM Agg:", output_batch_lstm)
