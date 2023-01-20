from pathlib import Path
import yaml

with open((Path(__file__).parent / "../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)


def create_moli_search_space(deactivate_triplet_loss):
    if deactivate_triplet_loss:
        gamma = {"name": "gamma", "value": 0, "value_type": "int", "type": "fixed"}
    else:
        gamma = {
            "name": "gamma",
            "values": parameter["gamma_choices"],
            "value_type": "float",
            "type": "choice",
        }
    search_space = [
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "h_dim1",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "h_dim2",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "h_dim3",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "lr_e",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_m",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_c",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "lr_cl",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_rate_e",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_rate_m",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_rate_c",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_rate_clf",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "weight_decay",
            "values": parameter["weight_decay_choices"],
            "log_scale": True,
            "value_type": "float",
            "type": "choice",
        },
        gamma,
        {
            "name": "margin",
            "values": parameter["margin_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "value_type": "int",
            "type": "range",
        },
    ]
    return search_space


def create_super_felt_search_space():
    search_space = [
        {
            "name": "encoder_dropout",
            "values": parameter["drop_rate_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "classifier_dropout",
            "values": parameter["drop_rate_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "classifier_weight_decay",
            "values": parameter["weight_decay_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "encoder_weight_decay",
            "values": parameter["weight_decay_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "learning_rate_e",
            "values": parameter["learning_rate_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "learning_rate_m",
            "values": parameter["learning_rate_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "learning_rate_c",
            "values": parameter["learning_rate_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "learning_rate_classifier",
            "values": parameter["learning_rate_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "e_epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "type": "range",
            "value_type": "int",
        },
        {
            "name": "m_epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "type": "range",
            "value_type": "int",
        },
        {
            "name": "c_epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "type": "range",
            "value_type": "int",
        },
        {
            "name": "classifier_epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "type": "range",
            "value_type": "int",
        },
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "margin",
            "values": parameter["margin_choices"],
            "type": "choice",
            "value_type": "float",
        },
        {
            "name": "e_dimension",
            "values": parameter["dim_choice"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "m_dimension",
            "values": parameter["dim_choice"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "c_dimension",
            "values": parameter["dim_choice"],
            "type": "choice",
            "value_type": "int",
        },
    ]

    return search_space


def create_early_integration_search_space(deactivate_triplet_loss):
    if deactivate_triplet_loss:
        gamma = {"name": "gamma", "value": 0, "value_type": "int", "type": "fixed"}
    else:
        gamma = {
            "name": "gamma",
            "values": parameter["gamma_choices"],
            "value_type": "float",
            "type": "choice",
        }
    search_space = [
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "h_dim",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "lr",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "dropout_rate",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "weight_decay",
            "values": parameter["weight_decay_choices"],
            "log_scale": True,
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "margin",
            "values": parameter["margin_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "value_type": "int",
            "type": "range",
        },
        gamma,
    ]
    return search_space


def create_stacking_search_space(deactivate_triplet_loss):
    if deactivate_triplet_loss:
        gamma = {"name": "gamma", "value": 0, "value_type": "int", "type": "fixed"}
    else:
        gamma = {
            "name": "gamma",
            "values": parameter["gamma_choices"],
            "value_type": "float",
            "type": "choice",
        }
    search_space = [
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "h_dim_e_encode",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "h_dim_m_encode",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "h_dim_c_encode",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "lr_e",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_m",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_c",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_clf",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "dropout_e",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_m",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_c",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout_clf",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "weight_decay",
            "values": parameter["weight_decay_choices"],
            "log_scale": True,
            "value_type": "float",
            "type": "choice",
        },
        gamma,
        {
            "name": "margin",
            "values": parameter["margin_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "value_type": "int",
            "type": "range",
        },
    ]
    return search_space


def create_moma_search_space(add_triplet_loss):
    if not add_triplet_loss:
        gamma = {"name": "gamma", "value": 0, "value_type": "int", "type": "fixed"}
    else:
        gamma = {
            "name": "gamma",
            "values": parameter["gamma_choices"],
            "value_type": "float",
            "type": "choice",
        }
    search_space = [
        gamma,
        {
            "name": "margin",
            "values": parameter["margin_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "h_dim_classifier",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "modules",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "lr_expression",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_mutation",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_cna",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_classifier",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "weight_decay",
            "values": parameter["weight_decay_choices"],
            "log_scale": True,
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "value_type": "int",
            "type": "range",
        },
    ]
    return search_space


def create_omi_embed_search_space(add_triplet_loss):
    if not add_triplet_loss:
        gamma = {"name": "gamma", "value": 0, "value_type": "int", "type": "fixed"}
    else:
        gamma = {
            "name": "gamma",
            "values": parameter["gamma_choices"],
            "value_type": "float",
            "type": "choice",
        }
    search_space = [
        gamma,
        {
            "name": "margin",
            "values": parameter["margin_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "type": "choice",
            "value_type": "int",
        },
        {
            "name": "latent_space_dim",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "lr_vae",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "lr_classifier",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "log_scale": True,
            "type": "choice",
        },
        {
            "name": "weight_decay",
            "values": parameter["weight_decay_choices"],
            "log_scale": True,
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "epochs_phase",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "value_type": "int",
            "type": "range",
        },
        {
            "name": "k_kl",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "k_embed",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "leaky_slope",
            "values": parameter["weight_decay_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dim_1B",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "dim_1A",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "dim_1C",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
        {
            "name": "class_dim_1",
            "values": parameter["dim_choice"],
            "value_type": "int",
            "type": "choice",
        },
    ]

    return search_space


def create_pca_search_space():
    search_space = [
        {
            "name": "variance_e",
            "values": parameter["variance_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "variance_m",
            "values": parameter["variance_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "variance_c",
            "values": parameter["variance_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "dropout",
            "values": parameter["drop_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "learning_rate",
            "values": parameter["learning_rate_choices"],
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "weight_decay",
            "values": parameter["weight_decay_choices"],
            "log_scale": True,
            "value_type": "float",
            "type": "choice",
        },
        {
            "name": "epochs",
            "bounds": [parameter["epoch_lower"], parameter["epoch_upper"]],
            "type": "range",
            "value_type": "int",
        },
        {
            "name": "mini_batch",
            "values": parameter["batch_size_choices"],
            "value_type": "int",
            "type": "choice",
        },
    ]

    return search_space
