import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def output_dim(self, in_dim: int) -> int:
        return in_dim  # moyenne : préserve la dimension d'entrée

    def aggregate(self, neighbours_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbours_features (torch.Tensor): Tensor de forme (batch_size, num_neighbors, feature_dim)

        Returns:
            torch.Tensor: Tensor de forme (batch_size, feature_dim), moyenne element-wise
        """
        return torch.mean(neighbours_features, dim=1)


class MaxPoolingAggregator(nn.Module):
    def __init__(self, in_features: int, out_features: int = None):
        """
        Args:
            in_features  : dimension des features d'entrée
            out_features : dimension de sortie (par défaut = in_features pour être
                           compatible avec la concaténation [h_u, h_N] de dim 2*hidden_dim)
        """
        super(MaxPoolingAggregator, self).__init__()
        if out_features is None:
            out_features = in_features
        self.fc = nn.Linear(in_features, out_features)

    def output_dim(self, in_dim: int) -> int:
        return self.fc.out_features

    def aggregate(self, neighbours_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbours_features (torch.Tensor): (batch_size, num_neighbors, feature_dim)

        Returns:
            torch.Tensor: (batch_size, out_features), max après passage par MLP + ReLU
        """
        out = F.relu(self.fc(neighbours_features))
        return torch.max(out, dim=1)[0]


class LSTMAggregator(nn.Module):
    def __init__(self, in_features: int, out_features: int = None):
        """
        Args:
            in_features  : dimension des features d'entrée
            out_features : dimension cachée du LSTM (par défaut = in_features)
        """
        super(LSTMAggregator, self).__init__()
        if out_features is None:
            out_features = in_features
        # batch_first=True : entrée (batch, seq, feat)
        self.lstm = nn.LSTM(in_features, out_features, batch_first=True)

    def output_dim(self, in_dim: int) -> int:
        return self.lstm.hidden_size

    def aggregate(self, neighbours_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbours_features (torch.Tensor): (batch_size, num_neighbors, feature_dim)

        Returns:
            torch.Tensor: (batch_size, out_features)
        """
        # Mélange des voisins (LSTM non permutation-invariant)
        idx = torch.randperm(neighbours_features.size(1), device=neighbours_features.device)
        shuffled = neighbours_features[:, idx, :]
        _, (hn, _) = self.lstm(shuffled)
        # hn shape: (num_layers=1, batch, hidden) -> on prend la dernière couche
        return hn[-1]