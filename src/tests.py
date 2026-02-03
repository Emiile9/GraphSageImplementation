"""Fichier pour tester les fonctions utils sur petites instances"""
import numpy as np
import random
import torch
from src.utils.utils import get_n_neighbours
from src.layers import MeanAggregator

#Features
features = np.array([
    [1.0, 0.1, 5.0],  # Nœud 0
    [0.2, 0.8, 1.0],  # Nœud 1
    [1.5, 0.3, 4.0],  # Nœud 2
    [0.1, 0.9, 0.0],  # Nœud 3
    [1.2, 0.2, 4.5],  # Nœud 4
    [0.0, 0.7, 0.5]   # Nœud 5
], dtype=np.float32)

#Adjacency list
adj = {
    0: [1, 2, 4],
    1: [0, 3, 5],
    2: [0, 4],
    3: [1, 5],
    4: [0, 2],
    5: [1, 3]
}

neighbours = get_n_neighbours(adj, 0, 2)
print("2 Neighbors of node 0:", neighbours)

neighbours_feats = features[neighbours]
print(neighbours_feats)
input_tensor = torch.FloatTensor(neighbours_feats).unsqueeze(0)
print(input_tensor)
MeanAggregator_instance = MeanAggregator()
print(MeanAggregator_instance.forward(input_tensor))
