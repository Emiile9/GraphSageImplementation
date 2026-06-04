from typing import List, Callable, Iterable, Union

import torch
import torch.nn as nn
import networkx as nx

from src.utils.utils import get_n_neighbours, get_n_features


class AlgoMiniBatch(nn.Module):
    def __init__(
        self,
        depth: int,
        weight_matrices: nn.ModuleList,
        sigma: Callable,
        agg_f: nn.ModuleList,
        normalize_output: bool = True,
        in_features: int = None,
    ):
        """
        Args:
            depth (int): Nombre de couches K
            weight_matrices (nn.ModuleList): W^k, liste des matrices de poids par couche.
            sigma (Callable): Fonction d'activation (ex: F.relu)
            agg_f (nn.ModuleList): Liste des agrégateurs pour chaque couche.
            normalize_output (bool): si True, L2-normalise la sortie FINALE seulement.
            in_features (int): dimension des features d'entrée (peut différer de hidden_dim).
                Si None, on suppose in_features == hidden_dim (W[0].out_features).
        """
        super().__init__()
        self.K = depth
        self.W = weight_matrices
        self.sigma = sigma
        self.agg_f = agg_f
        self.normalize_output = normalize_output
        hidden_dim = weight_matrices[0].out_features
        self.in_features = in_features if in_features is not None else hidden_dim

    @staticmethod
    def _to_python_nodes(nodes: Union[torch.Tensor, Iterable]) -> List[int]:
        if isinstance(nodes, torch.Tensor):
            return [int(x) for x in nodes.detach().cpu().tolist()]
        out = []
        for x in nodes:
            if isinstance(x, torch.Tensor):
                out.append(int(x.detach().cpu().item()))
            else:
                out.append(int(x))
        return out

    def forward_propagation(self, G: nx.Graph, nodes, sampling_size: List[int]) -> torch.Tensor:
        if len(sampling_size) != self.K:
            raise ValueError(f"sampling_size doit être de longueur {self.K}, reçu {len(sampling_size)}")

        nodes_list = self._to_python_nodes(nodes)
        device = next(self.parameters()).device
        hidden_dim = self.W[0].out_features

        # Construction des ensembles B_k (de K vers 0) et des voisinages échantillonnés
        B = {self.K: set(nodes_list)}
        sampled_adj = {}

        for k in range(self.K, 0, -1):
            B[k - 1] = set(B[k])
            for u in B[k]:
                neighbors = get_n_neighbours(G, u, sampling_size[k - 1])
                sampled_adj[(k, u)] = neighbors
                B[k - 1].update(neighbors)

        # Initialisation h^0 (features)
        h = {}
        h0 = {}
        for node in B[0]:
            feats = get_n_features(G, node)
            if isinstance(feats, torch.Tensor):
                t = feats.clone().detach().float()
            else:
                t = torch.tensor(feats, dtype=torch.float)
            t = t.to(device)
            if t.numel() == 0:
                t = torch.zeros(hidden_dim, device=device)
            h0[node] = t

        h[0] = h0

        # Propagation couche par couche
        for k in range(self.K):
            current_h = {}
            is_last_layer = (k == self.K - 1)
            # dimension d'entrée de l'agrégateur à cette couche
            in_dim_k = self.in_features if k == 0 else hidden_dim

            for u in B[k + 1]:
                neighbors = sampled_adj.get((k + 1, u), [])

                if len(neighbors) > 0:
                    neigh_feats = torch.stack([h[k][v] for v in neighbors], dim=0)
                    neigh_feats = neigh_feats.unsqueeze(0)  # (1, num_neigh, dim)
                    h_k_N = self.agg_f[k].aggregate(neigh_feats).view(-1)
                else:
                    agg_out = self.agg_f[k].output_dim(in_dim_k)
                    h_k_N = torch.zeros(agg_out, dtype=h[k][u].dtype, device=device)

                h_u_prev = h[k][u]
                concat = torch.cat((h_u_prev, h_k_N), dim=0)  # (2*hidden_dim,)
                h_u_k = self.W[k](concat.unsqueeze(0)).squeeze(0)
                h_u_k = self.sigma(h_u_k)

                # CORRECTION : on ne normalise PAS à chaque couche. Seulement en sortie,
                # et uniquement si normalize_output=True. Sinon ReLU+L2-norm combinés
                # détruisent le signal (vecteurs presque nuls après 2 couches).
                if is_last_layer and self.normalize_output:
                    norm = torch.norm(h_u_k, p=2)
                    if norm > 1e-8:
                        h_u_k = h_u_k / norm

                current_h[u] = h_u_k

            h[k + 1] = current_h

        z = torch.stack([h[self.K][u] for u in nodes_list], dim=0)
        return z