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


class PcaModel(nn.Module):
    def __init__(self, pca_e, pca_m, pca_c, classifier, device):
        super(PcaModel, self).__init__()
        self.pca_e = pca_e
        self.pca_m = pca_m
        self.pca_c = pca_c
        self.device = device
        self.classifier = classifier

    def forward(self, e, m, c):
        features_e = torch.FloatTensor(self.pca_e.transform(e.cpu().detach()))
        features_m = torch.FloatTensor(self.pca_m.transform(m.cpu().detach()))
        features_c = torch.FloatTensor(self.pca_c.transform(c.cpu().detach()))

        input = torch.concat([features_e, features_m, features_c], axis=1).to(
            self.device
        )
        return self.classifier(input)
