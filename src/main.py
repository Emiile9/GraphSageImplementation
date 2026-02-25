import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from Algo_Mini_Batch import AlgoMiniBatch
from train import train
from layers import MeanAggregator

# --------------------------------------------------
# 0. Device + Seeds
# --------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

print("Device :", device)

# --------------------------------------------------
# 1. Charger le graphe réel (.pt)
# --------------------------------------------------

G = torch.load(
    "GraphSageImplementation/src/generation_dataset/2026-02-10_14-21-11/graphs/ppi.pt",
    weights_only=False   # IMPORTANT pour PyTorch >= 2.6
)

print("\nGraph loaded")
print("Nb nodes :", G.number_of_nodes())
print("Nb edges :", G.number_of_edges())

# --------------------------------------------------
# 2. Vérification des features réelles
# --------------------------------------------------

node0 = list(G.nodes())[0]

print("\nKeys node :", G.nodes[node0].keys())

assert "x" in G.nodes[node0], "ERREUR : pas de features 'x'"

feat_dim = G.nodes[node0]["x"].shape[0]

print("Feature dimension :", feat_dim)
print("Example feature shape :", G.nodes[node0]["x"].shape)

# Vérifier cohérence
for node in list(G.nodes())[:10]:
    assert G.nodes[node]["x"].shape[0] == feat_dim

print("✔ Features cohérentes")

# --------------------------------------------------
# 3. Adapter au format attendu par ton modèle
# --------------------------------------------------

# Ton AlgoMiniBatch attend graph.nodes[node]["features"]
for node in G.nodes():
    G.nodes[node]["features"] = G.nodes[node]["x"]

print("✔ Features copiées vers clé 'features'")

# --------------------------------------------------
# 4. Définition du modèle GraphSAGE
# --------------------------------------------------

hidden_dim = feat_dim
depth = 2

W = nn.ModuleList([
    nn.Linear(2 * hidden_dim, hidden_dim)
    for _ in range(depth)
])

agg = [MeanAggregator() for _ in range(depth)]
sample_size = [5 for _ in range(depth)]

model = AlgoMiniBatch(depth, W, F.relu, agg).to(device)

print("\nNb paramètres :", sum(p.numel() for p in model.parameters()))

# --------------------------------------------------
# 5. Entraînement (1 epoch test)
# --------------------------------------------------

print("\n--------------------------------")
print("Training.")

train(
    model,
    G,
    device,
    sampling_size=sample_size,
    epochs=1,              #  1 seule epoch pour test
    learning_rate=3e-4,
    batch_size=32
)

print("\nTraining terminé.")