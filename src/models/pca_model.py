import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(Classifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.classifier(x)
