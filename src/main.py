##########  Imports  ##########

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from Algo_Mini_Batch import AlgoMiniBatch
from train import train
from layers import MeanAggregator, MaxPoolingAggregator, LSTMAggregator


##########  Arguments Terminal  ##########

parser = argparse.ArgumentParser()

parser.add_argument("--graph_path", type=str, default="./src/generation_dataset/2026-02-10_14-21-11/graphs/ppi.pt",
                    help="Chemin vers le fichier .pt du graphe")

parser.add_argument("--aggregator", type=str, default="mean",
                    choices=["mean", "max", "lstm"],
                    help="Type d'agrégateur")

parser.add_argument("--sample_size", type=int, default=5,
                    help="Taille d'échantillonnage")

parser.add_argument("--epochs", type=int, default=10,
                    help="Nombre d'époques")

parser.add_argument("--batch_size", type=int, default=128,
                    help="Taille de batch")

args = parser.parse_args()


##########  Reproductibilité  ##########

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print(f"Exemple de feature : {G.nodes[node0]['x']}")

feat_dim = G.nodes[node0]["x"].shape[0]

for node in G.nodes():
    G.nodes[node]["features"] = G.nodes[node]["x"]


##########  Choix agrégateur  ##########

depth = 2
hidden_dim = feat_dim

if args.aggregator == "mean":
    agg = [MeanAggregator() for _ in range(depth)]
elif args.aggregator == "max":
    agg = [MaxPoolingAggregator() for _ in range(depth)]
elif args.aggregator == "lstm":
    agg = [LSTMAggregator(hidden_dim) for _ in range(depth)]


##########  Modèle  ##########

W = nn.ModuleList([
    nn.Linear(2 * hidden_dim, hidden_dim)
    for _ in range(depth)
])

sample_size = [args.sample_size for _ in range(depth)]

model = AlgoMiniBatch(depth, W, F.relu, agg).to(device)

print("\nNb paramètres :", sum(p.numel() for p in model.parameters()))


##########  Entrainement  ##########

print("\n--------------------------------")
print("La phase d'entrainement débute..")

train(
    model,
    G,
    device,
    sampling_size=sample_size,
    epochs=args.epochs,
    learning_rate=3e-4,
    batch_size=args.batch_size
)

print("\nL'entrainement est terminé.")


##########  Extraction des embeddings  ##########

model.eval()

all_nodes = list(G.nodes())

with torch.no_grad():
    embeddings = model.forward_propagation(G, all_nodes, sample_size)

print("Shape embeddings :", embeddings.shape)
for i in range(50):
    print(embeddings[i])