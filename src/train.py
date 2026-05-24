import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.dataset import GraphSageDataset


def graphsage_unsupervised_loss(z_u, z_pos, z_neg):
    """
    Fonction de perte du papier GraphSage

    --- Paramètres ---
    z_u   : (batch, d) embeddings des noeuds centraux
    z_pos : (batch, d) embeddings des voisins positifs
    z_neg : (batch, Q, d) embeddings négatifs
    """

    # produit scalaire positif
    pos_score = torch.sum(z_u * z_pos, dim=1)
    pos_loss = F.logsigmoid(pos_score)

    # produit scalaire négatif
    neg_score = torch.bmm(z_neg, z_u.unsqueeze(2)).squeeze(2)  # produit scalaire par batch avec les bonnes dimensions
    neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # on somme sur les Q négatifs pour avoir une approximation de Q fois l'espérance (principe Monte-Carlo)

    loss = -(pos_loss + neg_loss).mean()
    return loss


def train(model, G, device, sampling_size, epochs=10, learning_rate=3e-4, batch_size=128, num_pairs=10000):
    """
    Entrainement d'un modèle GraphSage

    --- Paramètres ---
    :param model: le modèle à entrainer
    :param G: le graphe networkx sur lequel on souhaite obtenir les embeddings
    :param device: device pytorch
    :param sampling_size: tailles d'échantillonnage pour chaque couche
    :param epochs: nombre d'époques pour l'entrainement
    :param learning_rate: taux d'apprentissage
    :param batch_size: taille des batchs
    :param num_pairs: nombre de paires positives générées par random walks
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = GraphSageDataset(G, num_pairs=num_pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('--------------------------------')

    epoch_losses = []
    for epoch in range(epochs):
        print(f'\nEpoque {epoch+1} / {epochs}')

        losses = []

        pbar = tqdm(loader, leave=False)

        for u, pos, neg in pbar:

            u = u.tolist()
            pos = pos.tolist()
            neg = neg.tolist()

            z_u   = model.forward_propagation(G, u, sampling_size)
            z_pos = model.forward_propagation(G, pos, sampling_size)

            # Négatifs
            neg_flat = [n for row in neg for n in row]
            z_neg_flat = model.forward_propagation(G, neg_flat, sampling_size)

            d = z_u.shape[1]
            z_neg = z_neg_flat.view(len(u), -1, d)

            loss = graphsage_unsupervised_loss(z_u, z_pos, z_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            pbar.set_description(f"Batch loss: {loss_val:.4f}")

        avg = sum(losses) / len(losses)
        epoch_losses.append(avg)
        print(f'Average loss: {avg:.4f}')

    print('Entrainement terminé.')
    return epoch_losses

