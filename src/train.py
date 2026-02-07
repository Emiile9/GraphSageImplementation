import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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


def train(model, data_graph, device, epochs=100, batch_size=128):

    model.to(device)
    criterion = graphsage_unsupervised_loss
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    dataset = GraphSageDataset(data_graph.adj_list)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    print("--------------------------------")
    print("Training.")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")
        losses = []

        for u, pos, neg in dataloader:

            u = u.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            z = model(data_graph.x.to(device))

            z_u = z[u]
            z_pos = z[pos]
            z_neg = z[neg]

            loss = criterion(z_u, z_pos, z_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Average loss: {sum(losses) / len(losses):.4f}")

    print("Finished training.")
    print("--------------------------------")
