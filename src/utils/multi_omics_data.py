import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from utils.network_training_util import read_and_transpose_csv, feature_selection

def get_non_zero_variance_gen_indices(data):
    selector = VarianceThreshold(0)
    return selector.fit(data).get_support(indices=True)

def load_drug_data(data_path, drug, dataset, return_data_frames=False):
    drug = drug.split('_')[0]
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'
    expression_train = read_and_transpose_csv(expressions_homogenized_path
                                              / f'GDSC_exprs.{drug}.eb_with.{dataset}_exprs.{drug}.tsv')
    response_train = pd.read_csv(response_path / f"GDSC_response.{drug}.tsv",
                                 sep="\t", index_col=0, decimal=',')
    mutation_train = read_and_transpose_csv(sna_binary_path / f"GDSC_mutations.{drug}.tsv")
    cna_train = read_and_transpose_csv(cna_binary_path / f"GDSC_CNA.{drug}.tsv")
    cna_train = cna_train.loc[:, ~cna_train.columns.duplicated()]

    expression_extern = read_and_transpose_csv(expressions_homogenized_path /
                                               f"{dataset}_exprs.{drug}.eb_with.GDSC_exprs.{drug}.tsv")
    mutation_extern = read_and_transpose_csv(sna_binary_path / f"{dataset}_mutations.{drug}.tsv")
    cna_extern = read_and_transpose_csv(cna_binary_path / f"{dataset}_CNA.{drug}.tsv")
    cna_extern = cna_extern.loc[:, ~cna_extern.columns.duplicated()]
    response_extern = pd.read_csv(response_path / f"{dataset}_response.{drug}.tsv",
                                  sep="\t", index_col=0, decimal=',')

    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1
    response_train.rename(mapper=str, axis='index', inplace=True)
    response_extern.loc[response_extern.response == 'R'] = 0
    response_extern.loc[response_extern.response == 'S'] = 1
    response_extern.rename(mapper=str, axis='index', inplace=True)

    cna_extern = cna_extern.fillna(0)
    cna_extern[cna_extern != 0.0] = 1
    cna_train = cna_train.fillna(0)
    cna_train[cna_train != 0.0] = 1
    mutation_extern = mutation_extern.fillna(0)
    mutation_extern[mutation_extern != 0.0] = 1
    mutation_train = mutation_train.fillna(0)
    mutation_train[mutation_train != 0.0] = 1

    expression_train = expression_train[expression_train.columns[get_non_zero_variance_gen_indices(expression_train)]]
    mutation_train = mutation_train[mutation_train.columns[get_non_zero_variance_gen_indices(mutation_train)]]
    cna_train = cna_train[cna_train.columns[get_non_zero_variance_gen_indices(cna_train)]]

    expression_intersection_genes_index = expression_train.columns.intersection(expression_extern.columns)
    mutation_intersection_genes_index = mutation_train.columns.intersection(mutation_extern.columns)
    cna_intersection_genes_index = cna_train.columns.intersection(cna_extern.columns)

    extern_sample_intersection = expression_extern.index.intersection(mutation_extern.index)
    extern_sample_intersection = extern_sample_intersection.intersection(cna_extern.index)
    train_samples_intersection = expression_train.index.intersection(mutation_train.index)
    train_samples_intersection = train_samples_intersection.intersection(cna_train.index)

    expression_extern = expression_extern.loc[extern_sample_intersection, expression_intersection_genes_index]
    mutation_extern = mutation_extern.loc[extern_sample_intersection, mutation_intersection_genes_index]
    cna_extern = cna_extern.loc[extern_sample_intersection, cna_intersection_genes_index]
    response_extern = response_extern.loc[extern_sample_intersection, :]
    expression_train = expression_train.loc[train_samples_intersection, expression_intersection_genes_index]
    mutation_train = mutation_train.loc[train_samples_intersection, mutation_intersection_genes_index]
    cna_train = cna_train.loc[train_samples_intersection, cna_intersection_genes_index]
    response_train = response_train.loc[train_samples_intersection, :]

    y_train = response_train.response.to_numpy(dtype=int)
    y_extern = response_extern.response.to_numpy(dtype=int)
    if return_data_frames:
        return expression_train, mutation_train, cna_train, y_train, expression_extern, \
               mutation_extern, cna_extern, y_extern
    else:
        return expression_train.to_numpy(), mutation_train.to_numpy(), cna_train.to_numpy(), y_train, \
           expression_extern.to_numpy(), mutation_extern.to_numpy(), cna_extern.to_numpy(), y_extern


def load_drug_data_with_elbow(data_path, drug, dataset, return_data_frames=False):
    gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
                = load_drug_data(data_path, drug, dataset, True)

    gdsc_e, gdsc_m, gdsc_c = feature_selection(gdsc_e, gdsc_m, gdsc_c)
    expression_intersection_genes_index = gdsc_e.columns.intersection(extern_e.columns)
    mutation_intersection_genes_index = gdsc_m.columns.intersection(extern_m.columns)
    cna_intersection_genes_index = gdsc_c.columns.intersection(extern_c.columns)
    if return_data_frames:
        extern_e = extern_e.loc[:, expression_intersection_genes_index]
        extern_m = extern_m.loc[:, mutation_intersection_genes_index]
        extern_c = extern_c.loc[:, cna_intersection_genes_index]
    else:
        extern_e = extern_e.loc[:, expression_intersection_genes_index].to_numpy()
        extern_m = extern_m.loc[:, mutation_intersection_genes_index].to_numpy()
        extern_c = extern_c.loc[:, cna_intersection_genes_index].to_numpy()
        gdsc_e = gdsc_e.to_numpy()
        gdsc_m = gdsc_m.to_numpy()
        gdsc_c = gdsc_c.to_numpy()

    return gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r
