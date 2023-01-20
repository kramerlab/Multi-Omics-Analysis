import argparse


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    parser.add_argument('--deactivate_triplet_loss', action='store_true')
    parser.add_argument('--convert_ids', action='store_true')
    parser.add_argument('--add_triplet_loss', action='store_true')
    parser.add_argument('--stacking_type', default='less_stacking', choices=['all', 'less_stacking', 'only_single'])
    return parser.parse_args()
