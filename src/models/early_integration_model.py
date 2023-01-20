import torch
from torch import nn


class AdaptiveEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(AdaptiveEncoder, self).__init__()
        self.layer = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.layer(x)


class EarlyIntegration(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(EarlyIntegration, self).__init__()
        self.encoder = AdaptiveEncoder(input_size, output_size, dropout_rate)
        self.classify = nn.Linear(output_size, 1)

    def forward(self, concatenated):
        encoded = self.encoder(concatenated)
        return self.classify(encoded)

    def forward_with_features(self, concatenated):
        encoded = self.encoder(concatenated)
        return torch.squeeze(self.classify(encoded)), encoded
