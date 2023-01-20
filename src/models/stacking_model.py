import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.encoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

        self.decoder = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction

    def encode(self, x):
        return self.encoder(x)


class StackingModel(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, dropout, stacking_type):
        super(StackingModel, self).__init__()
        self.stacking_type = stacking_type
        self.expression_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.mutation_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.cna_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])

        e_encoding_size = encoding_sizes[0]
        m_encoding_size = encoding_sizes[1]
        c_encoding_size = encoding_sizes[2]
        em_encoding_size = e_encoding_size + m_encoding_size
        mc_encoding_size = m_encoding_size + c_encoding_size
        ec_encoding_size = e_encoding_size + c_encoding_size
        emc_encoding_size = e_encoding_size + m_encoding_size + c_encoding_size
        if stacking_type == "all":
            stacking_dimension = 7
        elif stacking_type == "less_stacking":
            stacking_dimension = 4
        else:
            stacking_dimension = 3

        self.e_classify = nn.Sequential(nn.Linear(e_encoding_size, 1), nn.Sigmoid())
        self.m_classify = nn.Sequential(nn.Linear(m_encoding_size, 1), nn.Sigmoid())
        self.c_classify = nn.Sequential(nn.Linear(c_encoding_size, 1), nn.Sigmoid())
        self.em_classify = nn.Sequential(nn.Linear(em_encoding_size, 1), nn.Sigmoid())
        self.mc_classify = nn.Sequential(nn.Linear(mc_encoding_size, 1), nn.Sigmoid())
        self.ec_classify = nn.Sequential(nn.Linear(ec_encoding_size, 1), nn.Sigmoid())
        self.emc_classify = nn.Sequential(nn.Linear(emc_encoding_size, 1), nn.Sigmoid())
        self.classify_all = nn.Linear(stacking_dimension, 1)

    def forward_with_features(self, expression, mutation, cna):
        encoded_e = self.expression_encoder(expression)
        encoded_m = self.mutation_encoder(mutation)
        encoded_c = self.cna_encoder(cna)

        classified_e = self.e_classify(encoded_e)
        classified_m = self.m_classify(encoded_m)
        classified_c = self.c_classify(encoded_c)

        if self.stacking_type == "less_stacking":
            classified_emc = self.emc_classify(
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)
            )
            classification = self.classify_all(
                torch.concat(
                    (classified_e, classified_m, classified_c, classified_emc), dim=1
                )
            )

        elif self.stacking_type == "all":
            classified_emc = self.emc_classify(
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)
            )
            classified_em = self.em_classify(
                torch.concat((encoded_e, encoded_m), dim=1)
            )
            classified_mc = self.mc_classify(
                torch.concat((encoded_m, encoded_c), dim=1)
            )
            classified_ec = self.ec_classify(
                torch.concat((encoded_e, encoded_c), dim=1)
            )
            classification = self.classify_all(
                torch.concat(
                    (
                        classified_e,
                        classified_m,
                        classified_c,
                        classified_em,
                        classified_mc,
                        classified_ec,
                        classified_emc,
                    ),
                    dim=1,
                )
            )

        else:
            classification = self.classify_all(
                torch.concat((classified_e, classified_m, classified_c), dim=1)
            )

        return [classification, torch.concat((encoded_e, encoded_m, encoded_c), dim=1)]

    def forward(self, expression, mutation, cna):
        encoded_e = self.expression_encoder(expression)
        encoded_m = self.mutation_encoder(mutation)
        encoded_c = self.cna_encoder(cna)

        classified_e = self.e_classify(encoded_e)
        classified_m = self.m_classify(encoded_m)
        classified_c = self.c_classify(encoded_c)

        if self.stacking_type == "less_stacking":
            classified_emc = self.emc_classify(
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)
            )
            classification = self.classify_all(
                torch.concat(
                    (classified_e, classified_m, classified_c, classified_emc), dim=1
                )
            )

        elif self.stacking_type == "all":
            classified_emc = self.emc_classify(
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)
            )
            classified_em = self.em_classify(
                torch.concat((encoded_e, encoded_m), dim=1)
            )
            classified_mc = self.mc_classify(
                torch.concat((encoded_m, encoded_c), dim=1)
            )
            classified_ec = self.ec_classify(
                torch.concat((encoded_e, encoded_c), dim=1)
            )
            classification = self.classify_all(
                torch.concat(
                    (
                        classified_e,
                        classified_m,
                        classified_c,
                        classified_em,
                        classified_mc,
                        classified_ec,
                        classified_emc,
                    ),
                    dim=1,
                )
            )

        else:
            classification = self.classify_all(
                torch.concat((classified_e, classified_m, classified_c), dim=1)
            )

        return classification
