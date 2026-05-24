##########  Imports  ##########

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from src.Algo_Mini_Batch import AlgoMiniBatch
from src.train import train
from src.layers import MeanAggregator, MaxPoolingAggregator, LSTMAggregator


##########  Arguments Terminal  ##########

parser = argparse.ArgumentParser()

parser.add_argument("--graph_path", type=str,
                    default="./src/generation_dataset/2026-02-10_14-21-11/graphs/ppi.pt",
                    help="Chemin vers le fichier .pt du graphe")

parser.add_argument("--aggregator", type=str, default="mean",
                    choices=["mean", "max", "lstm"],
                    help="Type d'agrégateur")

parser.add_argument("--sample_size", type=int, default=10,
                    help="Taille d'échantillonnage")

parser.add_argument("--epochs", type=int, default=30,
                    help="Nombre d'époques")

parser.add_argument("--batch_size", type=int, default=128,
                    help="Taille de batch")

parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")

parser.add_argument("--num_pairs", type=int, default=50000,
                    help="Nombre de paires positives générées par random walks")

args = parser.parse_args()


##########  Reproductibilité  ##########

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

print("Device :", device)


##########  Chargement du graphe  ##########

G = torch.load(args.graph_path, weights_only=False)

print("\nGraph loaded")
print("Nb nodes :", G.number_of_nodes())
print("Nb edges :", G.number_of_edges())

node0 = list(G.nodes())[0]
assert "x" in G.nodes[node0], "ERREUR : pas de features 'x'"

feat_dim = G.nodes[node0]["x"].shape[0]

for node in G.nodes():
    G.nodes[node]["features"] = G.nodes[node]["x"].float()


##########  Normalisation des features (z-score)  ##########
# IMPORTANT : sans ça, si les features sont toutes negatives,
# ReLU tue tout le signal et la loss reste bloquee a log(2)*(1+Q).

all_feats = torch.stack([G.nodes[n]["features"] for n in G.nodes()])
feat_mean = all_feats.mean(dim=0)
feat_std = all_feats.std(dim=0).clamp(min=1e-6)

for node in G.nodes():
    G.nodes[node]["features"] = (G.nodes[node]["features"] - feat_mean) / feat_std

check = torch.stack([G.nodes[n]["features"] for n in G.nodes()])
print(f"\nFeatures normalisees :")
print(f"  - moyenne : {check.mean().item():.4f}  (attendu ~ 0)")
print(f"  - std     : {check.std().item():.4f}  (attendu ~ 1)")
print(f"  - min / max : {check.min().item():.4f} / {check.max().item():.4f}")


##########  Choix agregateur  ##########

depth = 2
# hidden_dim fixé à 256 comme dans le papier (indépendant de feat_dim)
hidden_dim = 256

if args.aggregator == "mean":
    agg = nn.ModuleList([MeanAggregator()] +
                        [MeanAggregator() for _ in range(depth - 1)])
elif args.aggregator == "max":
    # couche 0 : feat_dim -> hidden_dim ; couches suivantes : hidden_dim -> hidden_dim
    agg = nn.ModuleList([MaxPoolingAggregator(feat_dim, hidden_dim)] +
                        [MaxPoolingAggregator(hidden_dim, hidden_dim) for _ in range(depth - 1)])
elif args.aggregator == "lstm":
    agg = nn.ModuleList([LSTMAggregator(feat_dim, hidden_dim)] +
                        [LSTMAggregator(hidden_dim, hidden_dim) for _ in range(depth - 1)])


##########  Modele  ##########

# W[0] adapte feat_dim -> hidden_dim ; W[1:] font hidden_dim -> hidden_dim
# Pour MeanAggregator : concat = feat_dim + feat_dim (mean préserve la dim)
# Pour Max/LSTM       : concat = feat_dim + hidden_dim (l'agrégateur projette)
agg_out_0 = agg[0].output_dim(feat_dim)
W = nn.ModuleList(
    [nn.Linear(feat_dim + agg_out_0, hidden_dim)] +
    [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(depth - 1)]
)

# Initialisation Kaiming adaptee a LeakyReLU
for layer in W:
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.1)
    nn.init.zeros_(layer.bias)

sample_size = [args.sample_size for _ in range(depth)]

# LeakyReLU au lieu de ReLU : evite le "dying ReLU problem"
activation = lambda x: F.leaky_relu(x, negative_slope=0.1)

model = AlgoMiniBatch(depth, W, activation, agg,
                      normalize_output=True, in_features=feat_dim).to(device)

print("\nNb parametres :", sum(p.numel() for p in model.parameters()))


##########  Entrainement  ##########

print("\n--------------------------------")
print("La phase d'entrainement debute..")

train(
    model,
    G,
    device,
    sampling_size=sample_size,
    epochs=args.epochs,
    learning_rate=args.lr,
    batch_size=args.batch_size,
    num_pairs=args.num_pairs,
)

print("\nL'entrainement est termine.")


##########  Extraction des embeddings  ##########

model.eval()

all_nodes = list(G.nodes())

with torch.no_grad():
    embeddings = model.forward_propagation(G, all_nodes, sample_size)

print("\nShape embeddings :", embeddings.shape)
print(f"Stats -- min: {embeddings.min().item():.4f}  "
      f"max: {embeddings.max().item():.4f}  "
      f"mean: {embeddings.mean().item():.4f}  "
      f"std: {embeddings.std().item():.4f}")
print(f"Nb dims non-nulles (>1e-6) moyen : "
      f"{(embeddings.abs() > 1e-6).float().sum(dim=1).mean().item():.1f} / {embeddings.shape[1]}")

for i in range(5):
    print(f"\nnoeud {i} :", embeddings[i])