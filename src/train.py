import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import GraphSageDataset


def graphsage_unsupervised_loss(z_u, z_pos, z_neg):
    """
    z_u   : (batch, d) embeddings des noeuds centraux
    z_pos : (batch, d) embeddings des voisins positifs
    z_neg : (batch, Q, d) embeddings négatifs
    """

    # produit scalaire positif
    pos_score = torch.sum(z_u * z_pos, dim=1)
    pos_loss = F.logsigmoid(pos_score)

    # produit scalaire négatif
    neg_score = torch.bmm(z_neg, z_u.unsqueeze(2)).squeeze(
        2
    )  # produit scalaire par batch avec les bonnes dimensions
    neg_loss = F.logsigmoid(-neg_score).sum(
        dim=1
    )  # on somme sur les Q négatifs pour avoir une approximation de Q fois l'espérance (principe Monte-Carlo)

    loss = -(pos_loss + neg_loss).mean()
    return loss


def train(model, G, device, sampling_size, epochs=100, learning_rate=3e-4, batch_size=128):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = GraphSageDataset(G)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('--------------------------------')
    print('Training.')

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1} / {epochs}')

        losses = []

        # 👉 barre de progression sur les batchs
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

            # 👉 mise à jour barre de progression
            pbar.set_description(f"Batch loss: {loss_val:.4f}")

        print(f'Average loss: {sum(losses)/len(losses):.4f}')

    print('Finished training.')

