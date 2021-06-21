import sys
from datetime import datetime
from pathlib import Path
import torch
import pandas as pd
import pickle
from ax import (
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    SearchSpace,
    SimpleExperiment,
    save,
    load,
    FixedParameter
)
from sklearn.model_selection import StratifiedKFold
from ax.modelbridge.registry import Models
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
import argparse
from pathlib import Path
import numpy as np
from training_bo_holi_moli import train_and_validate, train_final, test
from utils import multi_omics_data
from utils.visualisation import save_auroc_plots

mini_batch_list = [32, 64]
dim_list = [512, 256, 128, 64, 32, 16, 8]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
drop_rate_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.1, 0.01, 0.001, 0.0001]
gamma_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
combination_list = [0, 1, 2, 3, 4]
depth_list = [1, 2, 3, 4]
batch_size_list = [32, 64, 128]


def bo_moli(search_iterations, sobol_iterations, load_checkpoint, experiment_name, combination,
            sampling_method, drug_name, extern_dataset):
    if sampling_method == 'sobol':
        sobol_iterations = 0

    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")

    result_path = Path('..', '..', '..', 'results', drug_name, 'bayesian_optimisation', experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)
    file_mode = 'a' if load_checkpoint else 'w'
    result_file = open(result_path / 'logs.txt', file_mode)
    checkpoint_path = result_path / 'checkpoint.json'

    data_path = Path('..', '..', '..', 'data')
    if drug_name == 'egfr':
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r = multi_omics_data.load_egfr_data(data_path)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r = multi_omics_data.load_drug_data(data_path,
                                                                                                     drug_name,
                                                                                                     extern_dataset)
    moli_search_space = create_search_space(combination)

    random_seed = 42

    max_objective_list = []
    test_auc_list = []
    extern_auc_list = []
    now = datetime.now()
    result_file.write(f'Start experiment at {now}\n')
    cv_splits = 5
    skf = StratifiedKFold(n_splits=cv_splits, random_seed=random_seed)
    iteration = 0
    for train_index, test_index in tqdm(skf.split(gdsc_e, gdsc_r), total=skf.get_n_splits(), desc="k-fold"):

        x_train_e = gdsc_e[train_index]
        x_train_m = gdsc_m[train_index]
        x_train_c = gdsc_c[train_index]
        y_train = gdsc_r[train_index]
        x_test_e = gdsc_e[test_index]
        x_test_m = gdsc_m[test_index]
        x_test_c = gdsc_c[test_index]
        y_test = gdsc_r[test_index]
        result_file.write(f'\tIteration {iteration} with seed {random_seed}: \n')
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        sobol = Models.SOBOL(moli_search_space, seed=random_seed)

        # load or set up experiment with initial sobel runs
        if load_checkpoint & checkpoint_path.exists():
            print("Load checkpoint")
            experiment = load(str(checkpoint_path))
            max_objective = max(np.array([trial.objective_mean for trial in experiment.trials.values()]))
            experiment.evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                                         x_train_e, x_train_m,
                                                                                         x_train_c,
                                                                                         y_train,
                                                                                         device)
            print(f"Resuming after iteration {len(experiment.trials.values())}")
            if (result_path / 'best_parameters').exists():
                best_parameters = pickle.load(open(result_path / 'best_parameters', 'rb'))
            else:
                best_parameters = None
        else:
            best_parameters = None
            experiment = SimpleExperiment(
                name="BO-MOLI",
                search_space=moli_search_space,
                evaluation_function=lambda parameterization: train_and_validate(parameterization,
                                                                                x_train_e, x_train_m,
                                                                                x_train_c,
                                                                                y_train, device),
                objective_name="auroc",
                minimize=False,
            )

            print(f"Running Sobol initialization trials...")
            for i in range(sobol_iterations):
                print(f"Running Sobol initialisation {i + 1}/{sobol_iterations}")
                experiment.new_trial(generator_run=sobol.gen(1))
                experiment.eval()
            save(experiment, str(checkpoint_path))

        for i in range(len(experiment.trials.values()), search_iterations):
            print(f"Running GP+EI optimization trial {i + 1} ...")
            # Reinitialize GP+EI model at each step with updated data.
            if sampling_method == 'gp':
                gp_ei = Models.BOTORCH(experiment=experiment, data=experiment.fetch_data())
                generator_run = gp_ei.gen(1)
            else:
                generator_run = sobol.gen(1)

            experiment.new_trial(generator_run=generator_run)
            experiment.eval()
            max_objective = max(np.array([trial.objective_mean for trial in experiment.trials.values()]))
            experiment.evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                                         x_train_e, x_train_m,
                                                                                         x_train_c,
                                                                                         y_train,
                                                                                         device)
            save(experiment, str(checkpoint_path))

            if i % 10 == 0:
                best_parameters = extract_best_parameter(experiment)
                objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
                save_auroc_plots(objectives, result_path, sobol_iterations)
                print(best_parameters)

        # save results
        best_parameters = extract_best_parameter(experiment)
        objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
        save(experiment, str(checkpoint_path))
        pickle.dump(objectives, open(result_path / 'objectives', "wb"))
        pickle.dump(best_parameters, open(result_path / 'best_parameters', "wb"))
        save_auroc_plots(objectives, result_path, sobol_iterations)

        iteration += 1

        result_file.write(f'\t\t{str(best_parameters)}\n')

        model, scaler = train_final(best_parameters, x_train_e, x_train_m, x_train_c, y_train, device)
        auc_test = test(model, scaler, x_test_e, x_test_m, x_test_c, y_test, device)
        aux_extern = test(model, scaler, x_extern_e, x_extern_m, x_extern_c, y_extern, device)

        result_file.write(f'\t\tBest EGFR Test Auroc in iteration {iteration} = {max_objective}\n')
        result_file.write(f'\t\tEGFR Test Auroc = {auc_test}\n')
        result_file.write(f'\t\tEGFR Extern: AUROC = {aux_extern}\n')
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(aux_extern)

    print("Done!")

    result_dict = {
        'validation': max_objective_list,
        'test': auc_test_list,
        'extern': auc_test_extern_list
    }
    calculate_mean_and_std_auc(result_dict, result_file)
    result_file.close()


def extract_best_parameter(experiment):
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    best_parameters = best_arm.parameters
    return best_parameters


def calculate_mean_and_std_auc(result_dict, result_file):
    for result_name, result_value in result_dict.items():
        mean = np.mean(result_value)
        std = np.std(result_value)
        result_file.write(f'\t{result_name} mean: {mean}')
        result_file.write(f'\t{result_name} std: {std}')


def create_search_space(combination):
    if combination is None:
        combination_parameter = ChoiceParameter(name='combination', values=combination_list,
                                                parameter_type=ParameterType.INT)
    else:
        combination_parameter = FixedParameter(name='combination', value=combination,
                                               parameter_type=ParameterType.INT)
    return SearchSpace(
        parameters=[
            ChoiceParameter(name='mini_batch', values=batch_size_list, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim1", lower=8, upper=256, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim2", lower=8, upper=256, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim3", lower=8, upper=256, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim4", lower=8, upper=256, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim5", lower=8, upper=256, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_1", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_2", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_3", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_4", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_5", lower=1, upper=3, parameter_type=ParameterType.INT),
            ChoiceParameter(name="lr_e", values=learning_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="lr_m", values=learning_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="lr_c", values=learning_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="lr_cl", values=learning_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="lr_middle", values=learning_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="dropout_rate_e", values=drop_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="dropout_rate_m", values=drop_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="dropout_rate_c", values=drop_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="dropout_rate_clf", values=drop_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name="dropout_rate_middle", values=drop_rate_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name='weight_decay', values=weight_decay_list, parameter_type=ParameterType.FLOAT),
            ChoiceParameter(name='gamma', values=gamma_list, parameter_type=ParameterType.FLOAT),
            RangeParameter(name='epochs', lower=10, upper=100, parameter_type=ParameterType.INT),
            combination_parameter,
            ChoiceParameter(name='margin', values=margin_list, parameter_type=ParameterType.FLOAT),
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_iterations', default=1, type=int)
    parser.add_argument('--sobol_iterations', default=5, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--combination', default=None, type=int)
    parser.add_argument('--sampling_method', default='gp', choices=['gp', 'sobol'])
    args = parser.parse_args()

    drugs = {'gemcitabine_tcga': 'TCGA',
             'gemcitabine_pdx': 'PDX',
             'cisplatin': 'TCGA',
             'docetaxel': 'TCGA',
             'erlotinib': 'PDX',
             'cetuximab': 'PDX',
             'paclitaxel': 'PDX',
             'egfr': 'PDX'}
    for drug, extern_dataset in drugs.items():
        bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                args.combination, args.sampling_method, drug, extern_dataset)