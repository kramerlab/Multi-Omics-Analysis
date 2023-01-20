from ax import Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep

from utils.network_training_util import calculate_mean_and_std_auc


def create_generation_strategy():
    generation_strategy = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=-1, max_parallelism=5,
            model_gen_kwargs = {"optimizer_kwargs": {"joint_optimize": True}}),
        ],
        name="Sobol",
    )
    return generation_strategy


def write_results_to_file(
    drug_name,
    extern_auc_list,
    extern_auprc_list,
    result_file,
    test_auc_list,
    test_auprc_list,
):
    result_dict = {
        "test auroc": test_auc_list,
        "test auprc": test_auprc_list,
        "extern auroc": extern_auc_list,
        "extern auprc": extern_auprc_list,
    }
    result_file.write(f"\n test auroc list: {test_auc_list} \n")
    result_file.write(f"\n test auprc list: {test_auprc_list} \n")
    result_file.write(f"\n extern auroc list: {extern_auc_list} \n")
    result_file.write(f"\n extern auprc list: {extern_auprc_list} \n")
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
