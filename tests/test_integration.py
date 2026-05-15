"""
Test d'intégration : entraîne GraphSAGE sur un petit graphe synthétique
et vérifie les propriétés attendues des embeddings et de la loss.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from src.Algo_Mini_Batch import AlgoMiniBatch
from src.layers import MeanAggregator, MaxPoolingAggregator, LSTMAggregator
from src.train import train, graphsage_unsupervised_loss
from src.dataset import GraphSageDataset


# ---------------------------------------------------------------------------
# Graphe synthétique reproductible
# ---------------------------------------------------------------------------

def make_test_graph(n_nodes=100, seed=42):
    """Graphe de Barabási–Albert (connexe, degrés variés) avec features aléatoires."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    G = nx.barabasi_albert_graph(n_nodes, m=3, seed=seed)
    feat_dim = 16
    for node in G.nodes():
        G.nodes[node]["features"] = torch.randn(feat_dim)
    return G, feat_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(feat_dim, aggregator_type="mean", depth=2):
    hidden_dim = feat_dim

    if aggregator_type == "mean":
        agg = nn.ModuleList([MeanAggregator() for _ in range(depth)])
    elif aggregator_type == "max":
        agg = nn.ModuleList([MaxPoolingAggregator(hidden_dim, hidden_dim) for _ in range(depth)])
    elif aggregator_type == "lstm":
        agg = nn.ModuleList([LSTMAggregator(hidden_dim, hidden_dim) for _ in range(depth)])

    W = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(depth)])
    activation = lambda x: F.leaky_relu(x, negative_slope=0.1)
    model = AlgoMiniBatch(depth, W, activation, agg, normalize_output=True)
    return model


def run_training(G, model, epochs=5, batch_size=32, lr=1e-3, num_pairs=500, sample_size=5):
    device = torch.device("cpu")
    model.to(device)

    dataset = GraphSageDataset(G, num_pairs=num_pairs, walk_length=5,
                               context_size=2, num_neg=5)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampling_size = [sample_size, sample_size]

    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for u, pos, neg in loader:
            u = u.tolist()
            pos = pos.tolist()
            neg = neg.tolist()

            z_u   = model.forward_propagation(G, u, sampling_size)
            z_pos = model.forward_propagation(G, pos, sampling_size)

            neg_flat = [n for row in neg for n in row]
            z_neg_flat = model.forward_propagation(G, neg_flat, sampling_size)
            z_neg = z_neg_flat.view(len(u), -1, z_u.shape[1])

            loss = graphsage_unsupervised_loss(z_u, z_pos, z_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg)
        print(f"  Époque {epoch+1}/{epochs}  loss={avg:.4f}")

    return epoch_losses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """La loss doit baisser au cours de l'entraînement."""
    print("\n[test_loss_decreases]")
    G, feat_dim = make_test_graph()
    model = build_model(feat_dim, "mean")
    losses = run_training(G, model, epochs=5)

    first, last = losses[0], losses[-1]
    print(f"  loss initiale={first:.4f}  loss finale={last:.4f}")
    assert last < first, f"La loss n'a pas baissé : {first:.4f} → {last:.4f}"
    print("  ✓ loss décroissante")


def test_embeddings_normalized():
    """Avec normalize_output=True, chaque embedding doit avoir une norme ≈ 1."""
    print("\n[test_embeddings_normalized]")
    G, feat_dim = make_test_graph()
    model = build_model(feat_dim, "mean")
    model.eval()

    nodes = list(G.nodes())[:20]
    with torch.no_grad():
        z = model.forward_propagation(G, nodes, sampling_size=[5, 5])

    norms = z.norm(dim=1)
    print(f"  norme min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")
    assert (norms > 1e-6).all(), "Des embeddings ont une norme nulle"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Embeddings non normalisés : {norms}"
    print("  ✓ embeddings normalisés à 1")


def test_no_nan_in_embeddings():
    """Aucun NaN ni Inf dans les embeddings après entraînement."""
    print("\n[test_no_nan_in_embeddings]")
    G, feat_dim = make_test_graph()
    model = build_model(feat_dim, "mean")
    run_training(G, model, epochs=3)
    model.eval()

    nodes = list(G.nodes())
    with torch.no_grad():
        z = model.forward_propagation(G, nodes, sampling_size=[5, 5])

    assert not torch.isnan(z).any(), "NaN détectés dans les embeddings"
    assert not torch.isinf(z).any(), "Inf détectés dans les embeddings"
    print(f"  shape={z.shape}  min={z.min():.4f}  max={z.max():.4f}")
    print("  ✓ aucun NaN/Inf")


def test_all_aggregators_converge():
    """Mean, max-pool et LSTM doivent tous converger (loss finale < loss initiale)."""
    print("\n[test_all_aggregators_converge]")
    G, feat_dim = make_test_graph()

    for agg_type in ("mean", "max", "lstm"):
        model = build_model(feat_dim, agg_type)
        losses = run_training(G, model, epochs=4)
        drop = losses[0] - losses[-1]
        print(f"  {agg_type}: {losses[0]:.4f} → {losses[-1]:.4f}  (Δ={drop:.4f})")
        assert losses[-1] < losses[0], \
            f"Agrégateur '{agg_type}' : loss n'a pas baissé"
    print("  ✓ tous les agrégateurs convergent")


def test_similar_nodes_closer_embeddings():
    """
    Dans un graphe avec deux cliques séparées, les nœuds de la même clique
    doivent avoir des embeddings plus proches entre eux qu'avec l'autre clique.
    """
    print("\n[test_similar_nodes_closer_embeddings]")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    # Deux cliques de 15 nœuds, reliées par un seul pont
    G = nx.Graph()
    clique_a = list(range(15))
    clique_b = list(range(15, 30))
    G.add_edges_from([(i, j) for i in clique_a for j in clique_a if i < j])
    G.add_edges_from([(i, j) for i in clique_b for j in clique_b if i < j])
    G.add_edge(clique_a[-1], clique_b[0])  # pont unique

    feat_dim = 8
    for node in G.nodes():
        G.nodes[node]["features"] = torch.randn(feat_dim)

    model = build_model(feat_dim, "mean")
    run_training(G, model, epochs=8, num_pairs=300, sample_size=5, batch_size=16)
    model.eval()

    with torch.no_grad():
        z = model.forward_propagation(G, list(G.nodes()), sampling_size=[5, 5])

    z_a = z[clique_a]
    z_b = z[clique_b]

    # Similarité cosine moyenne intra-clique vs inter-clique
    sim_intra = (z_a @ z_a.T).mean().item()
    sim_inter = (z_a @ z_b.T).mean().item()

    print(f"  sim intra-clique={sim_intra:.4f}  sim inter-clique={sim_inter:.4f}")
    assert sim_intra > sim_inter, \
        f"Les nœuds de la même clique devraient être plus proches : {sim_intra:.4f} vs {sim_inter:.4f}"
    print("  ✓ embeddings intra-clique plus similaires qu'inter-clique")


if __name__ == "__main__":
    test_embeddings_normalized()
    test_no_nan_in_embeddings()
    test_loss_decreases()
    test_all_aggregators_converge()
    test_similar_nodes_closer_embeddings()
    print("\n=== Tous les tests d'intégration passent ===")
