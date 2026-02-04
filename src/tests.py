"""Fichier pour tester les fonctions utils sur petites instances"""

import numpy as np
import random
import torch
from src.utils.utils import get_n_neighbours
from src.layers import MeanAggregator, MaxPoolingAggregator, LSTMAggregator

# Features
features = np.array(
    [
        [1.0, 0.1, 5.0],  # Nœud 0
        [0.2, 0.8, 1.0],  # Nœud 1
        [1.5, 0.3, 4.0],  # Nœud 2
        [0.1, 0.9, 0.0],  # Nœud 3
        [1.2, 0.2, 4.5],  # Nœud 4
        [0.0, 0.7, 0.5],  # Nœud 5
    ],
    dtype=np.float32,
)

# Adjacency list
adj = {0: [1, 2, 4], 1: [0, 3, 5], 2: [0, 4], 3: [1, 5], 4: [0, 2], 5: [1, 3]}

MeanAggregator_instance = MeanAggregator()
MaxPoolingAggregator_instance = MaxPoolingAggregator(in_features=3, out_features=4)
LSTMAggregator_instance = LSTMAggregator(in_features=3, out_features=4)

# On veut l'embedding pour le noeud 0 et le noeud 1 simultanément
node_0_neighbors = get_n_neighbours(adj, 0, 2)  # ex: [1, 4]
node_1_neighbors = get_n_neighbours(adj, 1, 2)  # ex: [0, 5]

# Construction du batch : (2 nœuds, 2 voisins chacun, 3 features)
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
