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
from train_moli import train_final
from utils.visualisation import visualize_importances
from utils.choose_gpu import create_device

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)

best_hyperparameter = {
    "Cetuximab": {
        "mini_batch": 16,
        "h_dim1": 64,
        "h_dim2": 512,
        "h_dim3": 32,
        "lr_e": 0.01,
        "lr_m": 0.01,
        "lr_c": 0.001,
        "lr_cl": 0.001,
        "dropout_rate_e": 0.3,
        "dropout_rate_m": 0.7,
        "dropout_rate_c": 0.7,
        "dropout_rate_clf": 0.1,
        "weight_decay": 0.1,
        "gamma": 0.5,
        "margin": 0.5,
        "epochs": 6,
    },
    "Docetaxel": {
        "mini_batch": 8,
        "h_dim1": 1024,
        "h_dim2": 512,
        "h_dim3": 128,
        "lr_e": 0.01,
        "lr_m": 0.001,
        "lr_c": 0.01,
        "lr_cl": 0.001,
        "dropout_rate_e": 0.5,
        "dropout_rate_m": 0.5,
        "dropout_rate_c": 0.3,
        "dropout_rate_clf": 0.1,
        "weight_decay": 0.001,
        "gamma": 0.5,
        "margin": 0.2,
        "epochs": 3,
    },
    "Cisplatin": {
        "mini_batch": 8,
        "h_dim1": 512,
        "h_dim2": 256,
        "h_dim3": 256,
        "lr_e": 0.001,
        "lr_m": 0.01,
        "lr_c": 0.001,
        "lr_cl": 0.001,
        "dropout_rate_e": 0.3,
        "dropout_rate_m": 0.5,
        "dropout_rate_c": 0.7,
        "dropout_rate_clf": 0.3,
        "weight_decay": 0.05,
        "gamma": 0.3,
        "margin": 0.2,
        "epochs": 15,
    },
    "Erlotinib": {
        "mini_batch": 32,
        "h_dim1": 128,
        "h_dim2": 32,
        "h_dim3": 32,
        "lr_e": 0.001,
        "lr_m": 0.01,
        "lr_c": 0.001,
        "lr_cl": 0.01,
        "dropout_rate_e": 0.7,
        "dropout_rate_m": 0.3,
        "dropout_rate_c": 0.7,
        "dropout_rate_clf": 0.1,
        "weight_decay": 0.05,
        "gamma": 0.5,
        "margin": 0.2,
        "epochs": 7,
    },
    "Gemcitabine_pdx": {
        "mini_batch": 32,
        "h_dim1": 512,
        "h_dim2": 32,
        "h_dim3": 64,
        "lr_e": 0.01,
        "lr_m": 0.001,
        "lr_c": 0.001,
        "lr_cl": 0.001,
        "dropout_rate_e": 0.7,
        "dropout_rate_m": 0.5,
        "dropout_rate_c": 0.3,
        "dropout_rate_clf": 0.1,
        "weight_decay": 0.1,
        "gamma": 0.0,
        "margin": 0.5,
        "epochs": 16,
    },
    "Gemcitabine_tcga": {
        "mini_batch": 32,
        "h_dim1": 512,
        "h_dim2": 1024,
        "h_dim3": 256,
        "lr_e": 0.001,
        "lr_m": 0.01,
        "lr_c": 0.01,
        "lr_cl": 0.001,
        "dropout_rate_e": 0.5,
        "dropout_rate_m": 0.3,
        "dropout_rate_c": 0.5,
        "dropout_rate_clf": 0.7,
        "weight_decay": 0.1,
        "gamma": 0.5,
        "margin": 0.2,
        "epochs": 18,
    },
    "Paclitaxel": {
        "mini_batch": 32,
        "h_dim1": 256,
        "h_dim2": 32,
        "h_dim3": 128,
        "lr_e": 0.001,
        "lr_m": 0.01,
        "lr_c": 0.001,
        "lr_cl": 0.01,
        "dropout_rate_e": 0.7,
        "dropout_rate_m": 0.1,
        "dropout_rate_c": 0.7,
        "dropout_rate_clf": 0.7,
        "weight_decay": 0.05,
        "gamma": 0.3,
        "margin": 0.5,
        "epochs": 2,
    },
}

torch.manual_seed(parameter["random_seed"])
np.random.seed(parameter["random_seed"])


def moli_feature_importance(
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
        "moli",
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

    moli_model, scaler_gdsc = train_final(
        hyperparameter,
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        device,
        pin_memory=False,
    )
    moli_model.eval()

    gdsc_e_scaled = torch.Tensor(scaler_gdsc.fit_transform(gdsc_e)).to(device)
    gdsc_m = torch.FloatTensor(gdsc_m).to(device)
    gdsc_c = torch.FloatTensor(gdsc_c).to(device)

    extern_e_scaled = torch.Tensor(scaler_gdsc.transform(extern_e)).to(device)

    integradet_gradients = ShapleyValueSampling(moli_model)

    """ all_attributions_test = compute_importances_values_multiple_inputs(
        (gdsc_e_scaled, gdsc_m, gdsc_c),
        integradet_gradients,
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

    extern_e_scaled.requires_grad_()
    extern_m.requires_grad_()
    extern_c.requires_grad_()
    all_attributions_extern = compute_importances_values_multiple_inputs(
        (extern_e_scaled, extern_m, extern_c), integradet_gradients
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
            moli_feature_importance(
                args.experiment_name,
                drug,
                extern_dataset,
                args.convert_ids,
                args.gpu_number,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        moli_feature_importance(
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.convert_ids,
            args.gpu_number,
        )
