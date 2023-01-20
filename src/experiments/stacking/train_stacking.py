import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
from models.stacking_model import StackingModel
from utils.network_training_util import (
    get_loss_fn,
    create_data_loader,
    create_sampler,
    train,
    test,
)
from scipy.stats import sem

best_auroc = -1
cv_splits_inner = 5


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter(
    parameterization, x_e, x_m, x_c, y, device, pin_memory, stacking_type
):
    mini_batch = parameterization["mini_batch"]
    h_dim_e_encode = parameterization["h_dim_e_encode"]
    h_dim_m_encode = parameterization["h_dim_m_encode"]
    h_dim_c_encode = parameterization["h_dim_c_encode"]
    lr_e = parameterization["lr_e"]
    lr_m = parameterization["lr_m"]
    lr_c = parameterization["lr_c"]
    lr_clf = parameterization["lr_clf"]
    dropout_e = parameterization["dropout_e"]
    dropout_m = parameterization["dropout_m"]
    dropout_c = parameterization["dropout_c"]
    dropout_clf = parameterization["dropout_clf"]
    weight_decay = parameterization["weight_decay"]
    gamma = parameterization["gamma"]
    epochs = parameterization["epochs"]
    margin = parameterization["margin"]

    aucs_validate = []
    iteration = 1
    skf = StratifiedKFold(n_splits=cv_splits_inner)
    for train_index, validate_index in tqdm(
        skf.split(x_e, y), total=skf.get_n_splits(), desc="k-fold"
    ):
        x_train_e = x_e[train_index]
        x_train_m = x_m[train_index]
        x_train_c = x_c[train_index]
        y_train = y[train_index]

        x_validate_e = x_e[validate_index]
        x_validate_m = x_m[validate_index]
        x_validate_c = x_c[validate_index]
        y_validate = y[validate_index]

        scaler_gdsc = StandardScaler()
        x_train_e = scaler_gdsc.fit_transform(x_train_e)

        # Initialisation
        sampler = create_sampler(y_train)
        train_loader = create_data_loader(
            torch.FloatTensor(x_train_e),
            torch.FloatTensor(x_train_m),
            torch.FloatTensor(x_train_c),
            torch.FloatTensor(y_train),
            mini_batch,
            pin_memory,
            sampler,
        )

        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape

        loss_fn = get_loss_fn(margin, gamma)

        encoding_sizes = [h_dim_e_encode, h_dim_m_encode, h_dim_c_encode]
        input_sizes = [ie_dim, im_dim, ic_dim]
        dropout_rates = [dropout_e, dropout_m, dropout_c, dropout_clf]
        stacking_model = StackingModel(
            input_sizes, encoding_sizes, dropout_rates, stacking_type
        ).to(device)

        moli_optimiser = torch.optim.Adagrad(
            [
                {"params": stacking_model.expression_encoder.parameters(), "lr": lr_e},
                {"params": stacking_model.mutation_encoder.parameters(), "lr": lr_m},
                {"params": stacking_model.cna_encoder.parameters(), "lr": lr_c},
            ],
            lr=lr_clf,
            weight_decay=weight_decay,
        )
        for _ in trange(epochs, desc="Epoch"):
            train(train_loader, stacking_model, moli_optimiser, loss_fn, device, gamma)

        # validate
        auc_validate, _ = test(
            stacking_model,
            scaler_gdsc,
            x_validate_e,
            x_validate_m,
            x_validate_c,
            y_validate,
            device,
        )
        aucs_validate.append(auc_validate)

        if iteration < cv_splits_inner:
            open_folds = cv_splits_inner - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(
                np.concatenate([aucs_validate, remaining_best_results])
            )
            if check_best_auroc(best_possible_mean):
                print("Skip remaining folds.")
                break
        iteration += 1

    mean = np.mean(aucs_validate)
    set_best_auroc(mean)
    standard_error_of_mean = sem(aucs_validate)

    return {"auroc": (mean, standard_error_of_mean)}


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def set_best_auroc(new_auroc):
    global best_auroc
    if new_auroc > best_auroc:
        best_auroc = new_auroc


def train_final(
    parameterization,
    x_train_e,
    x_train_m,
    x_train_c,
    y_train,
    device,
    pin_memory,
    stacking_type,
):
    mini_batch = parameterization["mini_batch"]
    h_dim_e_encode = parameterization["h_dim_e_encode"]
    h_dim_m_encode = parameterization["h_dim_m_encode"]
    h_dim_c_encode = parameterization["h_dim_c_encode"]
    lr_e = parameterization["lr_e"]
    lr_m = parameterization["lr_m"]
    lr_c = parameterization["lr_c"]
    lr_clf = parameterization["lr_clf"]
    dropout_e = parameterization["dropout_e"]
    dropout_m = parameterization["dropout_m"]
    dropout_c = parameterization["dropout_c"]
    dropout_clf = parameterization["dropout_clf"]
    weight_decay = parameterization["weight_decay"]
    gamma = parameterization["gamma"]
    epochs = parameterization["epochs"]
    margin = parameterization["margin"]

    train_scaler_gdsc = StandardScaler()
    x_train_e = train_scaler_gdsc.fit_transform(x_train_e)

    _, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_c.shape

    loss_fn = get_loss_fn(margin, gamma)

    encoding_sizes = [h_dim_e_encode, h_dim_m_encode, h_dim_c_encode]
    input_sizes = [ie_dim, im_dim, ic_dim]
    dropout_rates = [dropout_e, dropout_m, dropout_c, dropout_clf]

    stacking_model = StackingModel(
        input_sizes, encoding_sizes, dropout_rates, stacking_type
    ).to(device)

    optimiser = torch.optim.Adagrad(
        [
            {"params": stacking_model.expression_encoder.parameters(), "lr": lr_e},
            {"params": stacking_model.mutation_encoder.parameters(), "lr": lr_m},
            {"params": stacking_model.cna_encoder.parameters(), "lr": lr_c},
        ],
        lr=lr_clf,
        weight_decay=weight_decay,
    )

    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight), replacement=True
    )
    train_loader = create_data_loader(
        torch.FloatTensor(x_train_e),
        torch.FloatTensor(x_train_m),
        torch.FloatTensor(x_train_c),
        torch.FloatTensor(y_train),
        mini_batch,
        pin_memory,
        sampler,
    )

    for _ in range(epochs):
        train(
            train_loader, stacking_model, optimiser, loss_fn, device, gamma
        )
    return stacking_model, train_scaler_gdsc
