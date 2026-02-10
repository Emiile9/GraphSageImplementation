"""
Fichier contenant l'algorithme forward propagation mini batch

Algo compètement fait par chatGPT pour debug.
"""

from typing import List, Callable, Iterable, Union

import torch
import torch.nn as nn
import networkx as nx

from utils.utils import get_n_neighbours, get_n_features


class AlgoMiniBatch(nn.Module):
    def __init__(
        self,
        depth: int,
        weight_matrices: nn.ModuleList,
        sigma: Callable,
        agg_f: nn.ModuleList,
    ):
        """
        Args:
            depth (int): Nombre de couches K
            weight_matrices (nn.ModuleList): W^k, liste des matrices de poids par couche.
                Chaque W[k] doit être de forme Linear(2*hidden_dim -> hidden_dim).
            sigma (Callable): Fonction d'activation (ex: F.relu)
            agg_f (nn.ModuleList): Liste des agrégateurs pour chaque couche.
                Chaque agrégateur expose une méthode .aggregate(neighbours_features)
                qui renvoie (batch, hidden_dim).
        """
        super().__init__()
        self.K = depth
        self.W = weight_matrices
        self.sigma = sigma
        self.agg_f = agg_f

    @staticmethod
    def _to_python_nodes(nodes: Union[torch.Tensor, Iterable]) -> List[int]:
        """
        Convertit un batch de noeuds (souvent Tensor PyTorch) en liste de int Python.
        Indispensable car NetworkX utilise des noeuds "int", et torch.Tensor(5) != 5.
        """
        if isinstance(nodes, torch.Tensor):
            # ex: tensor([1,2,3]) -> [1,2,3]
            return [int(x) for x in nodes.detach().cpu().tolist()]
        # cas DataLoader: liste/tuple de scalaires ou de tensors 0-d
        out = []
        for x in nodes:
            if isinstance(x, torch.Tensor):
                out.append(int(x.detach().cpu().item()))
            else:
                out.append(int(x))
        return out

    def forward_propagation(self, G: nx.Graph, nodes, sampling_size: List[int]) -> torch.Tensor:
        """
        Génère les embeddings pour un mini batch de noeuds.

        Args:
            G (nx.Graph): Graphe concerné (les features doivent être dans G.nodes[u]["features"])
            nodes: batch B de noeuds cibles (souvent torch.Tensor venant du DataLoader)
            sampling_size (List[int]): tailles d'échantillonnage pour chaque couche (longueur K)

        Returns:
            z (torch.Tensor): embeddings des noeuds du batch, shape (batch_size, hidden_dim)
        """
        if len(sampling_size) != self.K:
            raise ValueError(f"sampling_size doit être de longueur {self.K}, reçu {len(sampling_size)}")

        # Conversion critique: noeuds PyTorch -> int Python
        nodes_list = self._to_python_nodes(nodes)

        # Device du modèle (pour éviter des mélanges CPU/GPU)
        device = next(self.parameters()).device

        # Dimension interne (hidden_dim)
        hidden_dim = self.W[0].out_features

        # 1) Construction des ensembles de noeuds B_k et des voisinages échantillonnés
        B = {self.K: set(nodes_list)}
        sampled_adj = {}  # clé: (k, u) -> liste de voisins échantillonnés pour u à la couche k

        for k in range(self.K, 0, -1):
            B[k - 1] = set(B[k])
            for u in B[k]:
                neighbors = get_n_neighbours(G, u, sampling_size[k - 1])
                sampled_adj[(k, u)] = neighbors
                B[k - 1].update(neighbors)

        # 2) Initialisation h^0 (features)
        h = {}
        h0 = {}
        for node in B[0]:
            feats = get_n_features(G, node)

            # get_n_features peut renvoyer un Tensor (recommandé)
            if isinstance(feats, torch.Tensor):
                t = feats.clone().detach().float()
            else:
                t = torch.tensor(feats, dtype=torch.float)

            # Important: mettre sur le device du modèle
            t = t.to(device)

            # Sécurité: si features manquantes -> vecteur nul
            if t.numel() == 0:
                t = torch.zeros(hidden_dim, device=device)

            h0[node] = t

        h[0] = h0

        # 3) Propagation couche par couche
        for k in range(self.K):
            current_h = {}

            for u in B[k + 1]:
                neighbors = sampled_adj.get((k + 1, u), [])

                # a) Agrégation des voisins
                if len(neighbors) > 0:
                    neigh_feats = torch.stack([h[k][v] for v in neighbors], dim=0)  # (num_neigh, hidden_dim)
                    neigh_feats = neigh_feats.unsqueeze(0)  # (1, num_neigh, hidden_dim)
                    h_k_N = self.agg_f[k].aggregate(neigh_feats).view(-1)  # (hidden_dim,)
                else:
                    # cas sans voisin : vecteur nul de dimension hidden_dim
                    h_k_N = torch.zeros(hidden_dim, dtype=h[k][u].dtype, device=device)

                # Sécurité: s'assurer que h[k][u] a bien la dimension hidden_dim
                h_u_prev = h[k][u]
                if h_u_prev.numel() == 0:
                    h_u_prev = torch.zeros(hidden_dim, dtype=h_k_N.dtype, device=device)

                # b) Concaténation (doit être 2*hidden_dim)
                concat = torch.cat((h_u_prev, h_k_N), dim=0)  # (2*hidden_dim,)

                # c) Transformation linéaire + activation
                h_u_k = self.W[k](concat.unsqueeze(0)).squeeze(0)  # (hidden_dim,)
                h_u_k = self.sigma(h_u_k)

                # d) Normalisation L2 (optionnel)
                norm = torch.norm(h_u_k, p=2)
                if norm > 0:
                    h_u_k = h_u_k / norm

                current_h[u] = h_u_k

            h[k + 1] = current_h

        # 4) Sortie: embeddings des noeuds demandés (dans l'ordre du batch)
        z = torch.stack([h[self.K][u] for u in nodes_list], dim=0)
        return z
