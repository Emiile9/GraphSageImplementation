import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, neighbours_features):
        """
        Args:
            neighbors_features (torch.Tensor): Tensor de forme (batch_size, num_neighbors, feature_dim)

        Returns:
            torch.Tensor: Tensor de forme (batch_size, feature_dim), moyenne element wise
        """
        return torch.mean(neighbours_features, dim=1)
    
class MaxPoolingAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super(MaxPoolingAggregator, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, neighbours_features):
        """
        Args:
            neighbors_features (torch.Tensor): Tensor de forme (batch_size, num_neighbors, feature_dim)

        Returns:
            torch.Tensor: Tensor de forme (batch_size, feature_dim), max après passage par une couche linéaire
        """
        out = F.relu(self.fc(neighbours_features))
        return torch.max(out, dim=1)[0]