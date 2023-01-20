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
from train_super_felt import train_final
from utils.visualisation import visualize_importances
from utils.choose_gpu import create_device
from models.super_felt_model import SuperFelt

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)

best_hyperparameter = {
    "Cetuximab": {
        "encoder_dropout": 0.7,
        "classifier_dropout": 0.3,
        "classifier_weight_decay": 0.001,
        "encoder_weight_decay": 0.05,
        "learning_rate_e": 0.001,
        "learning_rate_m": 0.01,
        "learning_rate_c": 0.001,
        "learning_rate_classifier": 0.001,
        "e_epochs": 11,
        "m_epochs": 4,
        "c_epochs": 17,
        "classifier_epochs": 17,
        "mini_batch": 16,
        "margin": 0.5,
        "e_dimension": 512,
        "m_dimension": 32,
        "c_dimension": 1024,
    },
    "Docetaxel": {
        "encoder_dropout": 0.5,
        "classifier_dropout": 0.3,
        "classifier_weight_decay": 0.01,
        "encoder_weight_decay": 0.0001,
        "learning_rate_e": 0.01,
        "learning_rate_m": 0.001,
        "learning_rate_c": 0.001,
        "learning_rate_classifier": 0.001,
        "e_epochs": 3,
        "m_epochs": 6,
        "c_epochs": 12,
        "classifier_epochs": 15,
        "mini_batch": 32,
        "margin": 0.5,
        "e_dimension": 512,
        "m_dimension": 128,
        "c_dimension": 32,
    },
    "Cisplatin": {
        "encoder_dropout": 0.1,
        "classifier_dropout": 0.5,
        "classifier_weight_decay": 0.01,
        "encoder_weight_decay": 0.05,
        "learning_rate_e": 0.01,
        "learning_rate_m": 0.01,
        "learning_rate_c": 0.01,
        "learning_rate_classifier": 0.01,
        "e_epochs": 7,
        "m_epochs": 15,
        "c_epochs": 7,
        "classifier_epochs": 5,
        "mini_batch": 32,
        "margin": 0.2,
        "e_dimension": 256,
        "m_dimension": 32,
        "c_dimension": 1024,
    },
    "Erlotinib": {
        "encoder_dropout": 0.7,
        "classifier_dropout": 0.7,
        "classifier_weight_decay": 0.001,
        "encoder_weight_decay": 0.0001,
        "learning_rate_e": 0.01,
        "learning_rate_m": 0.001,
        "learning_rate_c": 0.01,
        "learning_rate_classifier": 0.001,
        "e_epochs": 18,
        "m_epochs": 5,
        "c_epochs": 18,
        "classifier_epochs": 3,
        "mini_batch": 16,
        "margin": 1.0,
        "e_dimension": 1024,
        "m_dimension": 256,
        "c_dimension": 256,
    },
    "Gemcitabine_pdx": {
        "encoder_dropout": 0.5,
        "classifier_dropout": 0.5,
        "classifier_weight_decay": 0.1,
        "encoder_weight_decay": 0.001,
        "learning_rate_e": 0.01,
        "learning_rate_m": 0.001,
        "learning_rate_c": 0.001,
        "learning_rate_classifier": 0.001,
        "e_epochs": 15,
        "m_epochs": 7,
        "c_epochs": 7,
        "classifier_epochs": 6,
        "mini_batch": 32,
        "margin": 0.5,
        "e_dimension": 64,
        "m_dimension": 64,
        "c_dimension": 64,
    },
    "Gemcitabine_tcga": {
        "encoder_dropout": 0.3,
        "classifier_dropout": 0.1,
        "classifier_weight_decay": 0.001,
        "encoder_weight_decay": 0.05,
        "learning_rate_e": 0.01,
        "learning_rate_m": 0.001,
        "learning_rate_c": 0.001,
        "learning_rate_classifier": 0.001,
        "e_epochs": 19,
        "m_epochs": 9,
        "c_epochs": 7,
        "classifier_epochs": 10,
        "mini_batch": 8,
        "margin": 1.0,
        "e_dimension": 1024,
        "m_dimension": 128,
        "c_dimension": 128,
    },
    "Paclitaxel": {
        "encoder_dropout": 0.7,
        "classifier_dropout": 0.3,
        "classifier_weight_decay": 0.05,
        "encoder_weight_decay": 0.0001,
        "learning_rate_e": 0.001,
        "learning_rate_m": 0.01,
        "learning_rate_c": 0.001,
        "learning_rate_classifier": 0.001,
        "e_epochs": 15,
        "m_epochs": 19,
        "c_epochs": 19,
        "classifier_epochs": 10,
        "mini_batch": 8,
        "margin": 0.5,
        "e_dimension": 64,
        "m_dimension": 64,
        "c_dimension": 32,
    },
}

torch.manual_seed(parameter["random_seed"])
np.random.seed(parameter["random_seed"])


def stacking_feature_importance(
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
        "super_felt",
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

    e_encoder, m_encoder, c_encoder, classifier, scaler_gdsc = train_final(
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        hyperparameter,
        device,
        False,
    )
    classifier.eval()

    gdsc_e_scaled = torch.Tensor(scaler_gdsc.fit_transform(gdsc_e)).to(device)
    gdsc_m = torch.FloatTensor(gdsc_m).to(device)
    gdsc_c = torch.FloatTensor(gdsc_c).to(device)

    extern_e_scaled = torch.Tensor(scaler_gdsc.transform(extern_e)).to(device)

    super_felt_model = SuperFelt(e_encoder, m_encoder, c_encoder, classifier)
    shapley = ShapleyValueSampling(super_felt_model)

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

    extern_e_scaled.requires_grad_()
    extern_m.requires_grad_()
    extern_c.requires_grad_()
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

    """  save_importance_results(
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
            stacking_feature_importance(
                args.experiment_name,
                drug,
                extern_dataset,
                args.convert_ids,
                args.gpu_number,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        stacking_feature_importance(
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.convert_ids,
            args.gpu_number,
        )
