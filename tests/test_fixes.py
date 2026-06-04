"""
Tests couvrant les 3 bugs corrigés :
  1. get_n_neighbours - sampling avec remise quand n > degré
  2. get_n_neighbours - seed numpy respectée
  3. dataset - node IDs réels (pas des indices)
"""

import random
import numpy as np
import torch
import networkx as nx
import pytest

from src.utils.utils import get_n_neighbours
from src.dataset import GraphSageDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph(edges, features=None):
    """Construit un graphe NetworkX simple avec des features optionnelles."""
    G = nx.Graph()
    G.add_edges_from(edges)
    feat_dim = 4
    for node in G.nodes():
        feat = features[node] if features else torch.randn(feat_dim)
        G.nodes[node]["features"] = feat
    return G


# ---------------------------------------------------------------------------
# Bug 1 : sampling avec remise quand n > degré
# ---------------------------------------------------------------------------

class TestGetNNeighbours:

    def test_returns_n_samples_when_degree_less_than_n(self):
        """Quand le nœud a moins de voisins que n, on doit quand même retourner n éléments (avec remise)."""
        G = make_graph([(0, 1), (0, 2)])  # nœud 0 a degré 2
        result = get_n_neighbours(G, 0, n=5)
        assert len(result) == 5, f"Attendu 5 éléments, obtenu {len(result)}"

    def test_all_returned_are_valid_neighbours(self):
        """Tous les éléments retournés doivent être de vrais voisins."""
        G = make_graph([(0, 1), (0, 2)])
        result = get_n_neighbours(G, 0, n=10)
        assert all(v in (1, 2) for v in result)

    def test_returns_n_samples_when_degree_equals_n(self):
        """Quand degré == n, retourner exactement n voisins."""
        G = make_graph([(0, 1), (0, 2), (0, 3)])
        result = get_n_neighbours(G, 0, n=3)
        assert len(result) == 3

    def test_returns_n_samples_when_degree_greater_than_n(self):
        """Quand degré > n, pas de remise nécessaire mais on doit avoir n éléments."""
        G = make_graph([(0, i) for i in range(1, 11)])  # degré 10
        result = get_n_neighbours(G, 0, n=5)
        assert len(result) == 5

    def test_returns_empty_for_isolated_node(self):
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]["features"] = torch.zeros(4)
        result = get_n_neighbours(G, 0, n=3)
        assert result == []

    def test_returns_empty_for_missing_node(self):
        G = make_graph([(0, 1)])
        result = get_n_neighbours(G, 99, n=3)
        assert result == []


# ---------------------------------------------------------------------------
# Bug 2 : seed numpy respectée
# ---------------------------------------------------------------------------

class TestSeedReproducibility:

    def test_same_seed_same_result(self):
        """Deux appels avec la même seed doivent retourner les mêmes voisins."""
        G = make_graph([(0, i) for i in range(1, 20)])

        np.random.seed(42)
        r1 = get_n_neighbours(G, 0, n=5)

        np.random.seed(42)
        r2 = get_n_neighbours(G, 0, n=5)

        assert r1 == r2, f"Résultats différents avec la même seed : {r1} vs {r2}"

    def test_different_seeds_likely_different_result(self):
        """Deux seeds différentes doivent (très probablement) donner des résultats différents."""
        G = make_graph([(0, i) for i in range(1, 20)])

        np.random.seed(1)
        r1 = get_n_neighbours(G, 0, n=5)

        np.random.seed(999)
        r2 = get_n_neighbours(G, 0, n=5)

        # Avec 19 voisins et n=5, la probabilité de collision est négligeable
        assert r1 != r2


# ---------------------------------------------------------------------------
# Bug 3 : GraphSageDataset utilise les vrais node IDs
# ---------------------------------------------------------------------------

class TestDatasetNodeIds:

    def make_non_contiguous_graph(self):
        """Graphe dont les node IDs ne sont PAS 0,1,2,..."""
        G = nx.Graph()
        # Node IDs : 10, 20, 30, 40, 50
        edges = [(10, 20), (20, 30), (30, 40), (40, 50), (10, 50), (20, 40)]
        G.add_edges_from(edges)
        for node in G.nodes():
            G.nodes[node]["features"] = torch.randn(4)
        return G

    def test_positive_pairs_contain_valid_node_ids(self):
        """u_nodes et pos_nodes doivent contenir des vrais node IDs du graphe."""
        G = self.make_non_contiguous_graph()
        valid_nodes = set(G.nodes())
        dataset = GraphSageDataset(G, num_pairs=50, walk_length=3)

        for node_id in dataset.u_nodes.tolist():
            assert node_id in valid_nodes, f"ID invalide dans u_nodes : {node_id}"
        for node_id in dataset.pos_nodes.tolist():
            assert node_id in valid_nodes, f"ID invalide dans pos_nodes : {node_id}"

    def test_negative_samples_contain_valid_node_ids(self):
        """Les négatifs retournés par __getitem__ doivent être de vrais node IDs."""
        G = self.make_non_contiguous_graph()
        valid_nodes = set(G.nodes())
        dataset = GraphSageDataset(G, num_pairs=20, walk_length=3, num_neg=5)

        for i in range(len(dataset)):
            _, _, neg = dataset[i]
            for node_id in neg.tolist():
                assert node_id in valid_nodes, f"Négatif invalide : {node_id}"

    def test_contiguous_graph_still_works(self):
        """Le cas habituel (IDs 0..N-1) doit continuer à fonctionner."""
        G = make_graph([(i, i + 1) for i in range(9)])
        valid_nodes = set(G.nodes())
        dataset = GraphSageDataset(G, num_pairs=30, walk_length=3)

        for node_id in dataset.u_nodes.tolist():
            assert node_id in valid_nodes

    def test_dataset_len(self):
        G = make_graph([(0, 1), (1, 2), (2, 3)])
        dataset = GraphSageDataset(G, num_pairs=50)
        assert len(dataset) == 50

    def test_getitem_shapes(self):
        G = make_graph([(i, i + 1) for i in range(9)])
        num_neg = 7
        dataset = GraphSageDataset(G, num_pairs=20, num_neg=num_neg)
        u, pos, neg = dataset[0]
        assert neg.shape == (num_neg,), f"Shape négatifs : {neg.shape}"


# ---------------------------------------------------------------------------
# Test d'intégration : forward_propagation avec graph non-contigu
# ---------------------------------------------------------------------------

class TestForwardPropagationNonContiguousGraph:

    def test_forward_propagation_non_contiguous_nodes(self):
        """Le modèle doit fonctionner même si les node IDs ne sont pas 0..N-1."""
        import torch.nn as nn
        import torch.nn.functional as F
        from src.Algo_Mini_Batch import AlgoMiniBatch
        from src.layers import MeanAggregator

        G = nx.Graph()
        edges = [(10, 20), (20, 30), (30, 40), (40, 50), (10, 50)]
        G.add_edges_from(edges)
        feat_dim = 4
        for node in G.nodes():
            G.nodes[node]["features"] = torch.randn(feat_dim)

        depth = 2
        W = nn.ModuleList([nn.Linear(2 * feat_dim, feat_dim) for _ in range(depth)])
        agg = nn.ModuleList([MeanAggregator() for _ in range(depth)])
        model = AlgoMiniBatch(depth, W, F.relu, agg, normalize_output=True)

        nodes = [10, 20, 30]
        z = model.forward_propagation(G, nodes, sampling_size=[2, 2])

        assert z.shape == (3, feat_dim)
        assert not torch.isnan(z).any()
