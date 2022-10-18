import argparse
import sys
import time

import yaml
import torch
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots
from utils.experiment_utils import write_results_to_file
from models.super_felt_model import Classifier
from utils import multi_omics_data
from utils.choose_gpu import get_free_gpu
from train_super_felt import optimise_super_felt_parameter, compute_super_felt_metrics

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)
best_auroc = 0


def super_felt(
    experiment_name,
    drug_name,
    extern_dataset_name,
    gpu_number,
    search_iterations,
    architecture,
):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
    random_seed = parameter["random_seed"]
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    sobol_iterations = search_iterations

    data_path = Path(file_directory, "..", "..", "..", "data")
    result_path = Path(
        file_directory, "..", "..", "..", "results", "super.felt", drug_name, experiment_name
    )
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / "results.txt", "w")
    (
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        extern_e,
        extern_m,
        extern_c,
        extern_r,
    ) = multi_omics_data.load_drug_data_with_elbow(
        data_path, drug_name, extern_dataset_name
    )

    classifier = Classifier

    test_auc_list = []
    extern_auc_list = []
    objectives_list = []
    test_auprc_list = []
    test_validation_list = []
    extern_auprc_list = []
    start_time = time.time()

    skf_outer = StratifiedKFold(
        n_splits=parameter["cv_splits"], random_state=random_seed, shuffle=True
    )
    iteration = 0
    for train_index_outer, test_index in tqdm(
        skf_outer.split(gdsc_e, gdsc_r),
        total=skf_outer.get_n_splits(),
        desc=" Outer k-fold",
    ):
        global best_auroc
        best_auroc = 0
        x_train_val_e = gdsc_e[train_index_outer]
        x_test_e = gdsc_e[test_index]
        x_train_val_m = gdsc_m[train_index_outer]
        x_test_m = gdsc_m[test_index]
        x_train_val_c = gdsc_c[train_index_outer]
        x_test_c = gdsc_c[test_index]
        y_train_val = gdsc_r[train_index_outer]
        y_test = gdsc_r[test_index]
        best_parameters, experiment = optimise_super_felt_parameter(
            search_iterations,
            x_train_val_e,
            x_train_val_m,
            x_train_val_c,
            y_train_val,
            device,
        )
        external_AUC, external_AUCPR, test_AUC, test_AUCPR = compute_super_felt_metrics(
            x_test_e,
            x_test_m,
            x_test_c,
            x_train_val_e,
            x_train_val_m,
            x_train_val_c,
            best_parameters,
            device,
            extern_e,
            extern_m,
            extern_c,
            extern_r,
            y_test,
            y_train_val,
        )

        test_auc_list.append(test_AUC)
        extern_auc_list.append(external_AUC)
        test_auprc_list.append(test_AUCPR)
        extern_auprc_list.append(external_AUCPR)
        objectives = np.array(
            [trial.objective_mean for trial in experiment.trials.values()]
        )
        save_auroc_plots(objectives, result_path, iteration, sobol_iterations)

        max_objective = max(
            np.array([trial.objective_mean for trial in experiment.trials.values()])
        )
        objectives_list.append(objectives)

        test_validation_list.append(max_objective)
        result_file.write(f"\t\tBest {drug_name} validation Auroc = {max_objective}\n")
        iteration += 1

    end_time = time.time()
    result_file.write(f"\tMinutes needed: {round((end_time - start_time) / 60)}")
    write_results_to_file(
        drug_name,
        extern_auc_list,
        extern_auprc_list,
        result_file,
        test_auc_list,
        test_auprc_list,
    )
    save_auroc_with_variance_plots(
        objectives_list, result_path, "final", sobol_iterations
    )
    positive_extern = np.count_nonzero(extern_r == 1)
    negative_extern = np.count_nonzero(extern_r == 0)
    no_skill_prediction_auprc = positive_extern / (positive_extern + negative_extern)
    result_file.write(
        f"\n No skill predictor extern AUPRC: {no_skill_prediction_auprc} \n"
    )
    result_file.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--gpu_number", type=int)
    parser.add_argument(
        "--drug",
        default="all",
        choices=[
            "Gemcitabine_tcga",
            "Gemcitabine_pdx",
            "Cisplatin",
            "Docetaxel",
            "Erlotinib",
            "Cetuximab",
            "Paclitaxel",
        ],
    )
    parser.add_argument("--search_iterations", default=200, type=int)
    parser.add_argument(
        "--architecture", default=None, choices=["supervised-ae", "supervised-e"]
    )

    args = parser.parse_args()

    if args.drug == "all":
        for drug, extern_dataset in parameter["drugs"].items():
            super_felt(
                args.experiment_name,
                drug,
                extern_dataset,
                args.gpu_number,
                args.search_iterations,
                args.architecture,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        super_felt(
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.gpu_number,
            args.search_iterations,
            args.architecture,
        )
