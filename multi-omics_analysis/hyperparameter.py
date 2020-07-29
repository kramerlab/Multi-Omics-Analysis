drugs_hyperparameters = ({'drug': 'paclitaxel',
                          'mini_batch': 64,
                          'h_dim1': 512,
                          'h_dim2': 256,
                          'h_dim3': 1024,
                          'lrE': 0.0005,
                          'lrM': 0.5,
                          'lrC': 0.5,
                          'lrCL': 0.5,
                          'dropout_rateE': 0.4,
                          'dropout_rateM': 0.4,
                          'dropout_rateC': 0.5,
                          'weight_decay': 0.0001,
                          'dropout_rateClf': 0.3,
                          'gamma': 0.6,
                          'epochs': 10,
                          'margin': 0.5,
                          'expression_train': 'GDSC_exprs.Paclitaxel.eb_with.PDX_exprs.Paclitaxel.tsv',
                          'response_train': 'GDSC_response.Paclitaxel.tsv',
                          'mutation_train': 'GDSC_mutations.Paclitaxel.tsv',
                          'cna_train': 'GDSC_CNA.Paclitaxel.tsv',
                          'expression_test': 'PDX_exprs.Paclitaxel.eb_with.GDSC_exprs.Paclitaxel.tsv',
                          'response_test': 'PDX_response.Paclitaxel.tsv',
                          'mutation_test': 'PDX_mutations.Paclitaxel.tsv',
                          'cna_test': 'PDX_CNA.Paclitaxel.tsv'},

                         {'drug': 'pdx_gemcitabine',
                          'mini_batch': 13,
                          'h_dim1': 256,
                          'h_dim2': 32,
                          'h_dim3': 64,
                          'lrE': 0.05,
                          'lrM': 0.00001,
                          'lrC': 0.0005,
                          'lrCL': 0.001,
                          'dropout_rateE': 0.4,
                          'dropout_rateM': 0.6,
                          'dropout_rateC': 0.3,
                          'weight_decay': 0.001,
                          'dropout_rateClf': 0.6,
                          'gamma': 0.3,
                          'epochs': 5,
                          'margin': 1.5,
                          'expression_train': 'GDSC_exprs.Gemcitabine.eb_with.PDX_exprs.Gemcitabine.tsv',
                          'response_train': 'GDSC_response.Gemcitabine.tsv',
                          'mutation_train': 'GDSC_mutations.Gemcitabine.tsv',
                          'cna_train': 'GDSC_CNA.Gemcitabine.tsv',
                          'expression_test': 'PDX_exprs.Gemcitabine.eb_with.GDSC_exprs.Gemcitabine.tsv',
                          'response_test': 'PDX_response.Gemcitabine.tsv',
                          'mutation_test': 'PDX_mutations.Gemcitabine.tsv',
                          'cna_test': 'PDX_CNA.Gemcitabine.tsv'},

                         {'drug': 'cetuximab',
                          'mini_batch': 30,
                          'h_dim1': 256,
                          'h_dim2': 512,
                          'h_dim3': 128,
                          'lrE': 0.0001,
                          'lrM': 0.0005,
                          'lrC': 0.0005,
                          'lrCL': 0.0005,
                          'dropout_rateE': 0.3,
                          'dropout_rateM': 0.8,
                          'dropout_rateC': 0.8,
                          'weight_decay': 0.01,
                          'dropout_rateClf': 0.4,
                          'gamma': 0.2,
                          'epochs': 10,
                          'margin': 2,
                          'expression_train': 'GDSC_exprs.Cetuximab.eb_with.PDX_exprs.Cetuximab.tsv',
                          'response_train': 'GDSC_response.Cetuximab.tsv',
                          'mutation_train': 'GDSC_mutations.Cetuximab.tsv',
                          'cna_train': 'GDSC_CNA.Cetuximab.tsv',
                          'expression_test': 'PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv',
                          'response_test': 'PDX_response.Cetuximab.tsv',
                          'mutation_test': 'PDX_mutations.Cetuximab.tsv',
                          'cna_test': 'PDX_CNA.Cetuximab.tsv'},

                         {'drug': 'erlotinib',
                          'mini_batch': 33,
                          'h_dim1': 64,
                          'h_dim2': 64,
                          'h_dim3': 64,
                          'lrE': 0.5,
                          'lrM': 0.5,
                          'lrC': 0.1,
                          'lrCL': 0.1,
                          'dropout_rateE': 0.5,
                          'dropout_rateM': 0.5,
                          'dropout_rateC': 0.5,
                          'weight_decay': 0.01,
                          'dropout_rateClf': 0.5,
                          'gamma': 0.6,
                          'epochs': 5,
                          'margin': 1,
                          'expression_train': 'GDSC_exprs.Erlotinib.eb_with.PDX_exprs.Erlotinib.tsv',
                          'response_train': 'GDSC_response.Erlotinib.tsv',
                          'mutation_train': 'GDSC_mutations.Erlotinib.tsv',
                          'cna_train': 'GDSC_CNA.Erlotinib.tsv',
                          'expression_test': 'PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv',
                          'response_test': 'PDX_response.Erlotinib.tsv',
                          'mutation_test': 'PDX_mutations.Erlotinib.tsv',
                          'cna_test': 'PDX_CNA.Erlotinib.tsv'},

                         {'drug': 'docetaxel',
                          'mini_batch': 8,
                          'h_dim1': 16,
                          'h_dim2': 16,
                          'h_dim3': 16,
                          'lrE': 0.0001,
                          'lrM': 0.0005,
                          'lrC': 0.0005,
                          'lrCL': 0.001,
                          'dropout_rateE': 0.5,
                          'dropout_rateM': 0.5,
                          'dropout_rateC': 0.5,
                          'weight_decay': 0.0001,
                          'dropout_rateClf': 0.5,
                          'gamma': 0.4,
                          'epochs': 10,
                          'margin': 0.5,
                          'expression_train': 'GDSC_exprs.Docetaxel.eb_with.TCGA_exprs.Docetaxel.tsv',
                          'response_train': 'GDSC_response.Docetaxel.tsv',
                          'mutation_train': 'GDSC_mutations.Docetaxel.tsv',
                          'cna_train': 'GDSC_CNA.Docetaxel.tsv',
                          'expression_test': 'TCGA_exprs.Docetaxel.eb_with.GDSC_exprs.Docetaxel.tsv',
                          'response_test': 'TCGA_response.Docetaxel.tsv',
                          'mutation_test': 'TCGA_mutations.Docetaxel.tsv',
                          'cna_test': 'TCGA_CNA.Docetaxel.tsv'},

                         {'drug': 'cisplatin',
                          'mini_batch': 15,
                          'h_dim1': 128,
                          'h_dim2': 128,
                          'h_dim3': 128,
                          'lrE': 0.05,
                          'lrM': 0.005,
                          'lrC': 0.005,
                          'lrCL': 0.0005,
                          'dropout_rateE': 0.5,
                          'dropout_rateM': 0.6,
                          'dropout_rateC': 0.8,
                          'weight_decay': 0.1,
                          'dropout_rateClf': 0.6,
                          'gamma': 0.2,
                          'epochs': 20,
                          'margin': 0.5,
                          'expression_train': 'GDSC_exprs.Cisplatin.eb_with.TCGA_exprs.Cisplatin.tsv',
                          'response_train': 'GDSC_response.Cisplatin.tsv',
                          'mutation_train': 'GDSC_mutations.Cisplatin.tsv',
                          'cna_train': 'GDSC_CNA.Cisplatin.tsv',
                          'expression_test': 'TCGA_exprs.Cisplatin.eb_with.GDSC_exprs.Cisplatin.tsv',
                          'response_test': 'TCGA_response.Cisplatin.tsv',
                          'mutation_test': 'TCGA_mutations.Cisplatin.tsv',
                          'cna_test': 'TCGA_CNA.Cisplatin.tsv'},

                         {'drug': 'tcga_gemcitabine',
                          'mini_batch': 13,
                          'h_dim1': 16,
                          'h_dim2': 16,
                          'h_dim3': 16,
                          'lrE': 0.001,
                          'lrM': 0.0001,
                          'lrC': 0.01,
                          'lrCL': 0.05,
                          'dropout_rateE': 0.5,
                          'dropout_rateM': 0.5,
                          'dropout_rateC': 0.5,
                          'weight_decay': 0.001,
                          'dropout_rateClf': 0.5,
                          'gamma': 0.6,
                          'epochs': 10,
                          'margin': 2,
                          'expression_train': 'GDSC_exprs.Gemcitabine.eb_with.TCGA_exprs.Gemcitabine.tsv',
                          'response_train': 'GDSC_response.Gemcitabine.tsv',
                          'mutation_train': 'GDSC_mutations.Gemcitabine.tsv',
                          'cna_train': 'GDSC_CNA.Gemcitabine.tsv',
                          'expression_test': 'TCGA_exprs.Gemcitabine.eb_with.GDSC_exprs.Gemcitabine.tsv',
                          'response_test': 'TCGA_response.Gemcitabine.tsv',
                          'mutation_test': 'TCGA_mutations.Gemcitabine.tsv',
                          'cna_test': 'TCGA_CNA.Gemcitabine.tsv'}
                         )