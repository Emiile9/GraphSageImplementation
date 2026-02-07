import torch
import random
from torch.utils.data import Dataset

class GraphSageDataset(Dataset):
    def __init__(
        self,
        adj_list,           # liste d'adjacence du graphe
        num_pairs=10000,    # nombre de paires positives à générer
        walk_length=5,      # longueur des random walks pour générer les paires positives
        context_size=2,     # taille du contexte pour les paires positives (nombre de voisins à gauche et à droite dans la random walk)
        num_neg=20,         # nombre de négatifs à échantillonner pour chaque paire positive
    ):
        self.adj_list = adj_list
        self.num_nodes = len(adj_list)
        self.num_pairs = num_pairs
        self.walk_length = walk_length
        self.context_size = context_size
        self.num_neg = num_neg

        # Distribution négative degree^(3/4)
        degrees = torch.tensor([len(neigh) for neigh in adj_list], dtype=torch.float)
        prob_neg = degrees.pow(0.75)
        prob_neg /= prob_neg.sum()

        self.prob_neg = prob_neg

        # Pré-génération des paires positives
        self.u_nodes, self.pos_nodes = self._generate_positive_pairs()


    def _generate_positive_pairs(self):
        u_nodes, pos_nodes = [], []

        # random walks pour générer les paires positives
        while len(u_nodes) < self.num_pairs:
            start = random.randrange(self.num_nodes)
            walk = [start]

            for _ in range(self.walk_length - 1):
                cur = walk[-1]
                if len(self.adj_list[cur]) == 0:
                    break
                walk.append(random.choice(self.adj_list[cur]))

            # Génération des paires positives à partir de la random walk
            for i, u in enumerate(walk):
                for j in range(
                    max(0, i - self.context_size),
                    min(len(walk), i + self.context_size + 1),
                ):
                    if i == j:
                        continue

                    u_nodes.append(u)
                    pos_nodes.append(walk[j])

                    if len(u_nodes) >= self.num_pairs:
                        break
                if len(u_nodes) >= self.num_pairs:
                    break

        return torch.tensor(u_nodes), torch.tensor(pos_nodes)


    def __len__(self):
        return self.num_pairs


    def __getitem__(self, idx):
        u = self.u_nodes[idx]
        pos = self.pos_nodes[idx]

        # Échantillonnage négatif aléatoire à chaque appel
        neg = torch.multinomial(
            self.prob_neg,
            self.num_neg,
            replacement=True,
        )

        return u, pos, neg