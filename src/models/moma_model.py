import torch
from torch import nn
import numpy as np


class Moma(nn.Module):
    def __init__(self, In_Nodes1, In_Nodes2, In_Nodes3, classifier_size, Modules):
        super(Moma, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()

        self.expression_FC1_x = nn.Linear(In_Nodes1, Modules, bias=False)
        self.expression_FC1_y = nn.Linear(In_Nodes1, Modules, bias=False)

        self.mutation_FC1_x = nn.Linear(In_Nodes2, Modules, bias=False)
        self.mutation_FC1_y = nn.Linear(In_Nodes2, Modules, bias=False)

        self.cna_FC1_x = nn.Linear(In_Nodes3, Modules, bias=False)
        self.cna_FC1_y = nn.Linear(In_Nodes3, Modules, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.expression_FC3 = nn.Sequential(nn.Linear(Modules * 4, 1), nn.Sigmoid())
        self.mutation_FC3 = nn.Sequential(nn.Linear(Modules * 4, 1), nn.Sigmoid())
        self.cna_FC3 = nn.Sequential(nn.Linear(Modules * 4, 1), nn.Sigmoid())

    def forward(self, expression, mutation, cna, with_features=False):
        expression_x = self.expression_FC1_x(expression)
        expression_y = self.expression_FC1_y(expression)

        mutation_x = self.mutation_FC1_x(mutation)
        mutation_y = self.mutation_FC1_y(mutation)

        cna_x = self.cna_FC1_x(cna)
        cna_y = self.cna_FC1_y(cna)

        expression = torch.cat(
            [
                expression_x.reshape(-1, 1, self.Modules),
                expression_y.reshape(-1, 1, self.Modules),
            ],
            dim=1,
        )
        mutation = torch.cat(
            [
                mutation_x.reshape(-1, 1, self.Modules),
                mutation_y.reshape(-1, 1, self.Modules),
            ],
            dim=1,
        )
        cna = torch.cat(
            [cna_x.reshape(-1, 1, self.Modules), cna_y.reshape(-1, 1, self.Modules)],
            dim=1,
        )

        features = torch.cat([expression, mutation, cna], dim=1)

        norm = torch.norm(expression, dim=1, keepdim=True)
        expression = expression.div(norm)

        norm = torch.norm(mutation, dim=1, keepdim=True)
        mutation = mutation.div(norm)

        norm = torch.norm(cna, dim=1, keepdim=True)
        cna = cna.div(norm)

        energy_expression_mutation = torch.bmm(
            expression.reshape(-1, 2, self.Modules).permute(0, 2, 1),
            mutation.reshape(-1, 2, self.Modules),
        )

        energy_expression_cna = torch.bmm(
            expression.reshape(-1, 2, self.Modules).permute(0, 2, 1),
            cna.reshape(-1, 2, self.Modules),
        )

        energy_mutation_cna = torch.bmm(
            mutation.reshape(-1, 2, self.Modules).permute(0, 2, 1),
            cna.reshape(-1, 2, self.Modules),
        )

        attention_expression_mutation = self.softmax(
            energy_expression_mutation.permute(0, 2, 1)
        ).permute(0, 2, 1)
        attention_expression_cna = self.softmax(
            energy_expression_cna.permute(0, 2, 1)
        ).permute(0, 2, 1)

        attention_mutation_expression = self.softmax(
            energy_expression_mutation
        ).permute(0, 2, 1)
        attention_mutation_cna = (
            self.softmax(energy_mutation_cna).permute(0, 2, 1).permute(0, 2, 1)
        )

        attention_cna_expression = self.softmax(energy_expression_cna).permute(0, 2, 1)
        attention_cna_mutation = self.softmax(energy_mutation_cna).permute(0, 2, 1)

        expression_mutation_weighted = torch.bmm(
            expression, attention_expression_mutation
        )
        expression_cna_weighted = torch.bmm(expression, attention_expression_cna)
        expression = torch.concat(
            [expression_mutation_weighted, expression_cna_weighted], dim=1
        )

        mutation_expression_weighted = torch.bmm(
            mutation, attention_mutation_expression
        )
        mutation_cna_weighted = torch.bmm(mutation, attention_mutation_cna)
        mutation = torch.concat(
            [mutation_expression_weighted, mutation_cna_weighted], dim=1
        )

        cna_expression_weighted = torch.bmm(cna, attention_cna_expression)
        cna_mutation_attention_weighted = torch.bmm(cna, attention_cna_mutation)
        cna = torch.concat(
            [cna_expression_weighted, cna_mutation_attention_weighted], dim=1
        )

        expression = expression.view(-1, self.Modules * 4)
        mutation = mutation.view(-1, self.Modules * 4)
        cna = cna.view(-1, self.Modules * 4)

        expression = self.expression_FC3(expression)
        mutation = self.mutation_FC3(mutation)
        cna = self.cna_FC3(cna)

        if with_features:
            return (
                torch.squeeze(expression),
                torch.squeeze(mutation),
                torch.squeeze(cna),
                features,
            )
        else:
            return (
                torch.squeeze(expression),
                torch.squeeze(mutation),
                torch.squeeze(cna),
            )


class FullMomaModel(nn.Module):
    def __init__(self, classifier, logistic_regression, device):
        super(FullMomaModel, self).__init__()
        self.classifier = classifier
        self.logistic_regression = logistic_regression
        self.device = device

    def forward(self, e, m, c):
        prediction_e, prediction_m, prediction_c = self.classifier(e, m, c)
        X = np.stack(
            [
                prediction_e.detach().cpu(),
                prediction_m.detach().cpu(),
                prediction_c.detach().cpu(),
            ],
            axis=-1,
        )
        X = np.nan_to_num(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        final_probabilities = self.logistic_regression.predict_proba(X)[:, 1]
        return torch.FloatTensor(final_probabilities).to(self.device)
