"""
Fichier contenant l'algorithme forward propagation mini batch
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from src.utils.utils import get_n_neighbours,get_n_features
from typing import List,Callable

class AlgoMiniBatch(nn.Module):
    def __init__(self, depth: int, weight_matrices: nn.ModuleList, sigma: Callable, agg_f: nn.ModuleList):
        """
        Args :
            depth (int) : Nombre de couches
            weight_matrices (nn.ModuleList) : W^k, liste des matrices de poids pour chaque couche
            sigma (Callable) : Fonction d'activation non linéaire 
            agg_f (nn.ModuleList) : Liste des fonctions d'agrégation pour chaque couche
        """
        super(AlgoMiniBatch, self).__init__()
        self.K = depth
        self.W = weight_matrices
        self.sigma = sigma
        self.agg_f = agg_f

    def forward_propagation(self, G: nx.Graph, nodes, sampling_size: List[int]) -> torch.Tensor:
        """
        Génère les embeddings pour un mini batch de noeuds

        Args:
            G (nx.Graph) : Graphe concerné
            nodes : B, Batch de noeuds cibles
            sampling_size (List[int]) : Taille du sampling pour chaque couche

        Returns: 
            z (torch.Tensor) : Embeddings pour le mini batch
        """

        B = {self.K: set(nodes)}
        sampled_adj = {}
        for k in range(self.K, 0, -1):
            B[k-1] = set(B[k])
            for u in B[k]:
                neighbors = get_n_neighbours(G, u, sampling_size[k-1])
                sampled_adj[(k, u)] = neighbors
                B[k-1].update(neighbors)

        h = {}
        h[0] = {node: torch.tensor(get_n_features(G, node)) for node in B[0]}

        for k in range(self.K):
            current_h = {}
            for u in B[k+1]:
                neighbors = sampled_adj.get((k+1, u), [])
                h_k_1_list = [h[k][v] for v in neighbors]
                
                if len(h_k_1_list) > 0:
                    h_k_1 = torch.stack(h_k_1_list).unsqueeze(0)
                    h_k_N = self.agg_f[k].aggregate(h_k_1)
                    h_k_N = h_k_N.view(-1)
                else:
                    feat_dim = h[k][u].shape[0] if k==0 else self.W[k-1].out_features
                    h_k_N = torch.zeros(feat_dim)

                concat = torch.cat((h[k][u], h_k_N), dim=0)
                
                h_u_k = self.sigma(self.W[k](concat))

                norm = torch.norm(h_u_k, p=2)
                if norm != 0:
                    h_u_k = h_u_k / norm
                current_h[u] = h_u_k
            
            h[k+1] = current_h

        z = torch.stack([h[self.K][u] for u in nodes])
        return z

