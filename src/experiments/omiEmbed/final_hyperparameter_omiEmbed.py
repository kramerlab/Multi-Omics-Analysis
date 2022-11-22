import sys
from pathlib import Path
import torch
import numpy as np
import yaml
from ax import optimize

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.experiment_utils import create_generation_strategy
from utils.input_arguments import get_cmd_arguments
from utils.searchspaces import create_omi_embed_search_space
from utils.choose_gpu import get_free_gpu
from train_omiEmbed import (
    optimise_hyperparameter,
    reset_best_auroc,
)
from utils import multi_omics_data

file_directory = Path(__file__).parent

with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)


def omiEmbed(
    search_iterations,
    experiment_name,
    drug_name,
    extern_dataset_name,
    gpu_number,
    add_triplet_loss,
):
    device, pin_memory = create_device(gpu_number)
    result_path = Path(
        file_directory,
        "..",
        "..",
        "..",
        "results",
        "omi_embed",
        drug_name,
        experiment_name,
    )
    result_path.mkdir(parents=True, exist_ok=True)

    result_file = open(result_path / "results.txt", "w")
    log_file = open(result_path / "logs.txt", "w")
    log_file.write(f"Start for {drug_name}\n")

    data_path = Path(file_directory, "..", "..", "..", "data")

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

    omi_embed_search_space = create_omi_embed_search_space(add_triplet_loss)

    torch.manual_seed(parameter["random_seed"])
    np.random.seed(parameter["random_seed"])

    result_file.write(f"\t{iteration = }. \n")

    reset_best_auroc()
    evaluation_function = lambda parameterization: optimise_hyperparameter(
        parameterization,
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        device,
        pin_memory,
    )
    generation_strategy = create_generation_strategy()

    best_parameters, _, experiment, _ = optimize(
        total_trials=search_iterations,
        experiment_name="Moma",
        objective_name="auroc",
        parameters=omi_embed_search_space,
        evaluation_function=evaluation_function,
        minimize=False,
        generation_strategy=generation_strategy,
    )

    # save results
    max_objective = max(
        np.array([trial.objective_mean for trial in experiment.trials.values()])
    )

    iteration += 1

    result_file.write(f"\t\t{str(best_parameters) = }\n")
    result_file.write(f"\t\tBest {drug_name} test Auroc = {max_objective}\n")

    print("Done!")
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
    best_arm_name = df.arm_name[df["mean"] == df["mean"].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    best_parameters = best_arm.parameters
    return best_parameters


if __name__ == "__main__":
    args = get_cmd_arguments()
    if args.drug == "all":
        for drug, extern_dataset in parameter["drugs"].items():
            omiEmbed(
                args.search_iterations,
                args.experiment_name,
                drug,
                extern_dataset,
                args.gpu_number,
                args.add_triplet_loss,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        omiEmbed(
            args.search_iterations,
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.gpu_number,
            args.add_triplet_loss,
        )
