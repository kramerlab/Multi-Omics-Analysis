import yaml
import torch
from pathlib import Path
import numpy as np
import sys
from captum.attr import ShapleyValueSampling

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.input_arguments import get_cmd_arguments
from utils import multi_omics_data
from utils.interpretability import (
    compute_importances_values_multiple_inputs,
    save_importance_results,
)
from train_pca import train_final
from models.pca_model import PcaModel
from utils.visualisation import visualize_importances
from utils.choose_gpu import create_device


file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)

best_hyperparameter = {
    "Cetuximab": {
        "variance_e": 0.99,
        "variance_m": 0.9,
        "variance_c": 0.95,
        "dropout": 0.1,
        "learning_rate": 0.01,
        "weight_decay": 0.1,
        "epochs": 8,
        "mini_batch": 8,
    },
    "Docetaxel": {
        "variance_e": 0.9,
        "variance_m": 0.975,
        "variance_c": 0.99,
        "dropout": 0.3,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "epochs": 9,
        "mini_batch": 32,
    },
    "Cisplatin": {
        "variance_e": 0.95,
        "variance_m": 0.975,
        "variance_c": 0.975,
        "dropout": 0.7,
        "learning_rate": 0.01,
        "weight_decay": 0.05,
        "epochs": 19,
        "mini_batch": 16,
    },
    "Erlotinib": {
        "variance_e": 0.9,
        "variance_m": 0.9,
        "variance_c": 0.95,
        "dropout": 0.7,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "epochs": 14,
        "mini_batch": 32,
    },
    "Gemcitabine_pdx": {
        "variance_e": 0.95,
        "variance_m": 0.9,
        "variance_c": 0.99,
        "dropout": 0.7,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "epochs": 6,
        "mini_batch": 8,
    },
    "Gemcitabine_tcga": {
        "variance_e": 0.95,
        "variance_m": 0.99,
        "variance_c": 0.975,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "epochs": 6,
        "mini_batch": 16,
    },
    "Paclitaxel": {
        "variance_e": 0.99,
        "variance_m": 0.95,
        "variance_c": 0.99,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 5,
        "mini_batch": 8,
    },
}

torch.manual_seed(parameter["random_seed"])
np.random.seed(parameter["random_seed"])


def pca_feature_importance(
    experiment_name, drug_name, extern_dataset_name, convert_ids, gpu_number
):
    hyperparameter = best_hyperparameter[drug_name]
    device, _ = create_device(gpu_number)
    result_path = Path(
        file_directory,
        "..",
        "..",
        "..",
        "results",
        "pca",
        "explanation",
        experiment_name,
        drug_name,
    )
    result_path.mkdir(exist_ok=True, parents=True)
    data_path = Path(file_directory, "..", "..", "..", "data")
    (
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        extern_e,
        extern_m,
        extern_c,
        _,
    ) = multi_omics_data.load_drug_data_with_elbow(
        data_path, drug_name, extern_dataset_name, return_data_frames=True
    )
    # get columns names
    expression_columns = gdsc_e.columns
    expression_columns = [
        f"Expression {expression_gene}" for expression_gene in expression_columns
    ]

    mutation_columns = gdsc_m.columns
    mutation_columns = [
        f"Mutation {mutation_gene}" for mutation_gene in mutation_columns
    ]

    cna_columns = gdsc_c.columns
    cna_columns = [f"CNA {cna_gene}" for cna_gene in cna_columns]

    all_columns = np.concatenate([expression_columns, mutation_columns, cna_columns])

    gdsc_e = gdsc_e.to_numpy()
    gdsc_m = gdsc_m.to_numpy()
    gdsc_c = gdsc_c.to_numpy()
    extern_e = extern_e.to_numpy()
    extern_m = torch.FloatTensor(extern_m.to_numpy()).to(device)
    extern_c = torch.FloatTensor(extern_c.to_numpy()).to(device)

    number_of_expression_features = gdsc_e.shape[1]
    number_of_mutation_features = gdsc_m.shape[1]

    pca_model, train_scaler_gdsc, pca_e, pca_m, pca_c = train_final(
        hyperparameter,
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        device,
        pin_memory=False,
    )
    pca_model.eval()

    gdsc_e_scaled = torch.Tensor(train_scaler_gdsc.fit_transform(gdsc_e)).to(device)
    gdsc_m = torch.FloatTensor(gdsc_m).to(device)
    gdsc_c = torch.FloatTensor(gdsc_c).to(device)

    extern_e_scaled = torch.Tensor(train_scaler_gdsc.transform(extern_e)).to(device)

    full_pca_model = PcaModel(pca_e, pca_m, pca_c, pca_model, device)

    shapley = ShapleyValueSampling(full_pca_model)

    """ all_attributions_test = compute_importances_values_multiple_inputs(
        (gdsc_e_scaled, gdsc_m, gdsc_c),
        shapley,
    )

    visualize_importances(
        all_columns,
        all_attributions_test,
        path=result_path,
        file_name="all_attributions_test",
        convert_ids=convert_ids,
        number_of_expression_features=number_of_expression_features,
        number_of_mutation_features=number_of_mutation_features,
    ) """

    all_attributions_extern = compute_importances_values_multiple_inputs(
        (extern_e_scaled, extern_m, extern_c), shapley
    )

    visualize_importances(
        all_columns,
        all_attributions_extern,
        path=result_path,
        file_name="all_attributions_extern",
        convert_ids=convert_ids,
        number_of_expression_features=number_of_expression_features,
        number_of_mutation_features=number_of_mutation_features,
    )

    """ save_importance_results(
        all_attributions_test,
        all_columns,
        result_path,
        "extern",
    ) """
    save_importance_results(
        all_attributions_extern,
        all_columns,
        result_path,
        "test",
    )


if __name__ == "__main__":
    args = get_cmd_arguments()

    if args.drug == "all":
        for drug, extern_dataset in parameter["drugs"].items():
            pca_feature_importance(
                args.experiment_name,
                drug,
                extern_dataset,
                args.convert_ids,
                args.gpu_number,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        pca_feature_importance(
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.convert_ids,
            args.gpu_number,
        )
