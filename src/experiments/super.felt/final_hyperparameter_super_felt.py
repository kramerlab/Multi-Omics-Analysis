import sys
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import multi_omics_data
from utils.choose_gpu import get_free_gpu
from train_super_felt import optimise_super_felt_parameter
from utils.input_arguments import get_cmd_arguments

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
    deactivate_triplet_loss,
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

    data_path = Path(file_directory, "..", "..", "..", "data")
    result_path = Path(
        file_directory,
        "..",
        "..",
        "..",
        "results",
        "super.felt",
        drug_name,
        experiment_name,
    )
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / "results.txt", "w")
    (
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        _,
        _,
        _,
        _,
    ) = multi_omics_data.load_drug_data_with_elbow(
        data_path, drug_name, extern_dataset_name
    )

    global best_auroc
    best_auroc = 0
    best_parameters, experiment = optimise_super_felt_parameter(
        search_iterations,
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        device,
        deactivate_triplet_loss,
    )

    max_objective = max(
        np.array([trial.objective_mean for trial in experiment.trials.values()])
    )

    result_file.write(f"\t\tBest {drug_name} validation Auroc = {max_objective}\n")
    result_file.write(f"\t\t{str(best_parameters) = }\n")

    print("Done!")


if __name__ == "__main__":
    args = get_cmd_arguments()

    if args.drug == "all":
        for drug, extern_dataset in parameter["drugs"].items():
            super_felt(
                args.experiment_name,
                drug,
                extern_dataset,
                args.gpu_number,
                args.search_iterations,
                args.deactivate_triplet_loss,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        super_felt(
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.gpu_number,
            args.search_iterations,
            args.deactivate_triplet_loss,
        )
