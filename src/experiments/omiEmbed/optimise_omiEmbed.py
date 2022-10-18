import sys
from datetime import datetime
from pathlib import Path
import torch
import pickle
import time
import numpy as np
import yaml
from tqdm import tqdm
from ax import optimize
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.experiment_utils import create_generation_strategy
from utils.input_arguments import get_cmd_arguments
from utils.searchspaces import create_omi_embed_search_space
from utils.choose_gpu import get_free_gpu
from training_omiEmbed import train_final, optimise_hyperparameter, reset_best_auroc, test_omi_embed
from utils import multi_omics_data
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots
from utils.network_training_util import calculate_mean_and_std_auc
file_directory = Path(__file__).parent

with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)


def bo_moli(search_iterations, experiment_name, drug_name, extern_dataset_name, gpu_number):
    device, pin_memory = create_device(gpu_number)
    result_path = Path(file_directory, '..', '..', '..', 'results', 'omi_embed', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    result_file = open(result_path / 'results.txt', 'w')
    log_file = open(result_path / 'logs.txt', 'w')
    log_file.write(f"Start for {drug_name}\n")

    data_path = Path(file_directory, '..', '..', '..', 'data')

    gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
        = multi_omics_data.load_drug_data_with_elbow(data_path, drug_name, extern_dataset_name)

    omi_embed_search_space = create_omi_embed_search_space()

    torch.manual_seed(parameter['random_seed'])
    np.random.seed(parameter['random_seed'])

    max_objective_list = []
    test_auc_list = []
    extern_auc_list = []
    test_auprc_list = []
    extern_auprc_list = []
    objectives_list = []
    now = datetime.now()
    result_file.write(f'Start experiment at {now}\n')
    skf = StratifiedKFold(n_splits=parameter['cv_splits'], random_state=parameter['random_seed'], shuffle=True)
    iteration = 0

    start_time = time.time()
    for train_index, test_index in tqdm(skf.split(gdsc_e, gdsc_r), total=skf.get_n_splits(), desc="Outer k-fold"):
        result_file.write(f'\t{iteration = }. \n')
        x_train_validate_e = gdsc_e[train_index]
        x_train_validate_m = gdsc_m[train_index]
        x_train_validate_c = gdsc_c[train_index]
        y_train_validate = gdsc_r[train_index]
        x_test_e = gdsc_e[test_index]
        x_test_m = gdsc_m[test_index]
        x_test_c = gdsc_c[test_index]
        y_test = gdsc_r[test_index]

        reset_best_auroc()
        evaluation_function = lambda parameterization: optimise_hyperparameter(parameterization,
                                                                               x_train_validate_e,
                                                                               x_train_validate_m,
                                                                               x_train_validate_c,
                                                                               y_train_validate, device, pin_memory)
        generation_strategy = create_generation_strategy()

        best_parameters, values, experiment, model = optimize(
            total_trials=search_iterations,
            experiment_name='Moma',
            objective_name='auroc',
            parameters=omi_embed_search_space,
            evaluation_function=evaluation_function,
            minimize=False,
            generation_strategy=generation_strategy
        )

        # save results
        max_objective = max(np.array([trial.objective_mean for trial in experiment.trials.values()]))
        objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
        pickle.dump(objectives, open(result_path / 'objectives', "wb"))
        pickle.dump(best_parameters, open(result_path / 'best_parameters', "wb"))
        save_auroc_plots(objectives, result_path, iteration, search_iterations)

        iteration += 1

        result_file.write(f'\t\t{str(best_parameters) = }\n')

        model_final, scaler_final = train_final(best_parameters, x_train_validate_e, x_train_validate_m,
                                                x_train_validate_c, y_train_validate, device,
                                                pin_memory)
        auc_test, auprc_test = test_omi_embed(model_final, scaler_final, x_test_e, x_test_m, x_test_c, y_test)
        auc_extern, auprc_extern = test_omi_embed(model_final, scaler_final, extern_e, extern_m, extern_c,
                                                  extern_r)

        result_file.write(f'\t\tBest {drug_name} validation Auroc = {max_objective}\n')
        objectives_list.append(objectives)
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(auc_extern)
        test_auprc_list.append(auprc_test)
        extern_auprc_list.append(auprc_extern)

    print("Done!")
    end_time = time.time()
    result_file.write(f'\tMinutes needed: {round((end_time - start_time) / 60)}')
    result_dict = {
        'validation auroc': max_objective_list,
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
    }
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
    save_auroc_with_variance_plots(objectives_list, result_path, 'final', search_iterations)
    positive_extern = np.count_nonzero(extern_r == 1)
    negative_extern = np.count_nonzero(extern_r == 0)
    no_skill_prediction_auprc = positive_extern / (positive_extern + negative_extern)
    result_file.write(f'\n No skill predictor extern AUPRC: {no_skill_prediction_auprc} \n')

    result_file.write(f'\n test auroc list: {test_auc_list} \n')
    result_file.write(f'\n test auprc list: {test_auprc_list} \n')
    result_file.write(f'\n extern auroc list: {extern_auc_list} \n')
    result_file.write(f'\n extern auprc list: {extern_auprc_list} \n')
    result_file.write(f'\n validation auroc list: {max_objective_list} \n')

    result_file.close()


def create_device(gpu_number):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
        pin_memory = False
    else:
        device = torch.device("cpu")
        pin_memory = False
    return device, pin_memory


def extract_best_parameter(experiment):
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    best_parameters = best_arm.parameters
    return best_parameters


if __name__ == '__main__':
    args = get_cmd_arguments()
    if args.drug == 'all':
        for drug, extern_dataset in parameter['drugs'].items():
            bo_moli(args.search_iterations, args.experiment_name, drug, extern_dataset, args.gpu_number)
    else:
        extern_dataset = parameter['drugs'][args.drug]
        bo_moli(args.search_iterations, args.experiment_name, args.drug, extern_dataset, args.gpu_number)
