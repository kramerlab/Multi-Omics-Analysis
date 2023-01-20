import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
from models.moma_model import Moma
from siamese_triplet.utils import AllTripletSelector
from utils.network_training_util import create_sampler, create_data_loader
from scipy.stats import sem
from sklearn.linear_model import LogisticRegression

best_auroc = -1
cv_splits_inner = 5


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter(parameterization, x_e, x_m, x_c, y, device, pin_memory):
    mini_batch = parameterization["mini_batch"]
    h_dim_classifier = parameterization["h_dim_classifier"]
    modules = parameterization["modules"]
    lr_expression = parameterization["lr_expression"]
    lr_mutation = parameterization["lr_mutation"]
    lr_cna = parameterization["lr_cna"]
    lr_classifier = parameterization["lr_classifier"]
    weight_decay = parameterization["weight_decay"]
    epochs = parameterization["epochs"]
    gamma = parameterization["gamma"]
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

        loss_fn = torch.nn.BCELoss()
        e_in = x_train_e.shape[-1]
        m_in = x_train_m.shape[-1]
        c_in = x_train_c.shape[-1]
        moma_model = Moma(e_in, m_in, c_in, h_dim_classifier, modules).to(device)

        moma_optimiser = torch.optim.Adagrad(
            [
                {
                    "params": moma_model.expression_FC1_x.parameters(),
                    "lr": lr_expression,
                },
                {
                    "params": moma_model.expression_FC1_y.parameters(),
                    "lr": lr_expression,
                },
                {"params": moma_model.mutation_FC1_x.parameters(), "lr": lr_mutation},
                {"params": moma_model.mutation_FC1_y.parameters(), "lr": lr_mutation},
                {"params": moma_model.cna_FC1_x.parameters(), "lr": lr_cna},
                {"params": moma_model.cna_FC1_y.parameters(), "lr": lr_cna},
                {"params": moma_model.expression_FC3.parameters(), "lr": lr_classifier},
                {"params": moma_model.mutation_FC3.parameters(), "lr": lr_classifier},
                {"params": moma_model.cna_FC3.parameters(), "lr": lr_classifier},
            ],
            weight_decay=weight_decay,
        )

        for _ in trange(epochs, desc="Epoch"):
            train_moma(
                train_loader, moma_model, moma_optimiser, loss_fn, device, gamma, margin
            )

        with torch.no_grad():
            moma_model = moma_model.cpu()
            expression_logit, mutation_logit, cna_logit = moma_model.forward(
                torch.FloatTensor(x_train_e),
                torch.FloatTensor(x_train_m),
                torch.FloatTensor(x_train_c),
            )
        X = np.stack([expression_logit, mutation_logit, cna_logit], axis=-1)
        logistic_regression = LogisticRegression().fit(X, y_train)

        # validate
        moma_model = moma_model.to(device)
        auc_validate, _ = test_moma(
            moma_model,
            scaler_gdsc,
            torch.FloatTensor(x_validate_e),
            torch.FloatTensor(x_validate_m),
            torch.FloatTensor(x_validate_c),
            y_validate,
            device,
            logistic_regression,
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
    parameterization, x_train_e, x_train_m, x_train_c, y_train, device, pin_memory
):
    mini_batch = parameterization["mini_batch"]
    h_dim_classifier = parameterization["h_dim_classifier"]
    modules = parameterization["modules"]
    lr_expression = parameterization["lr_expression"]
    lr_mutation = parameterization["lr_mutation"]
    lr_cna = parameterization["lr_cna"]
    lr_classifier = parameterization["lr_classifier"]
    weight_decay = parameterization["weight_decay"]
    epochs = parameterization["epochs"]
    gamma = parameterization["gamma"]
    margin = parameterization["margin"]

    train_scaler_gdsc = StandardScaler()
    train_scaler_gdsc.fit(x_train_e)
    x_train_e = train_scaler_gdsc.transform(x_train_e)

    loss_fn = torch.nn.BCELoss()

    e_in = x_train_e.shape[-1]
    m_in = x_train_m.shape[-1]
    c_in = x_train_c.shape[-1]
    moma_model = Moma(e_in, m_in, c_in, h_dim_classifier, modules).to(device)

    moma_optimiser = torch.optim.Adagrad(
        [
            {"params": moma_model.expression_FC1_x.parameters(), "lr": lr_expression},
            {"params": moma_model.expression_FC1_y.parameters(), "lr": lr_expression},
            {"params": moma_model.mutation_FC1_x.parameters(), "lr": lr_mutation},
            {"params": moma_model.mutation_FC1_y.parameters(), "lr": lr_mutation},
            {"params": moma_model.cna_FC1_x.parameters(), "lr": lr_cna},
            {"params": moma_model.cna_FC1_y.parameters(), "lr": lr_cna},
            {"params": moma_model.expression_FC3.parameters(), "lr": lr_classifier},
            {"params": moma_model.mutation_FC3.parameters(), "lr": lr_classifier},
            {"params": moma_model.cna_FC3.parameters(), "lr": lr_classifier},
        ],
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
        train_moma(
            train_loader,
            moma_model,
            moma_optimiser,
            loss_fn,
            device,
            gamma,
            margin,
        )

    with torch.no_grad():
        moma_model = moma_model.cpu()
        expression_logit, mutation_logit, cna_logit = moma_model.forward(
            torch.FloatTensor(x_train_e),
            torch.FloatTensor(x_train_m),
            torch.FloatTensor(x_train_c),

        )
    moma_model = moma_model.to(device)
    X = np.stack([expression_logit, mutation_logit, cna_logit], axis=-1)
    logistic_regression = LogisticRegression().fit(X, y_train)
    return moma_model, train_scaler_gdsc, logistic_regression


def train_moma(
    train_loader,
    model,
    optimiser,
    loss_fn,
    device,
    gamma,
    margin,
):
    y_true = []
    model.train()

    if gamma > 0:
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)
        triplet_selector = AllTripletSelector()
    else:
        triplet_loss_fn = None
        triplet_selector = None

    for (data_e, data_m, data_c, target) in train_loader:
        if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
            optimiser.zero_grad()
            y_true.extend(target)

            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)
            expression_logit, mutation_logit, cna_logit, features = model.forward(
                data_e, data_m, data_c, True
            )
            loss = (
                loss_fn(torch.squeeze(expression_logit), target)
                + loss_fn(torch.squeeze(mutation_logit), target)
                + loss_fn(torch.squeeze(cna_logit), target)
            )
            if gamma > 0:
                triplets = triplet_selector.get_triplets(features, target)
                triplet_loss = triplet_loss_fn(
                    features[triplets[:, 0], :],
                    features[triplets[:, 1], :],
                    features[triplets[:, 2], :],
                )
                loss += triplet_loss
            loss.backward()
            optimiser.step()


def test_moma(
    model, scaler, expression, mutation, cna, response, device, logistic_regression
):
    model = model.cpu()
    expression = torch.FloatTensor(scaler.transform(expression))
    mutation = torch.FloatTensor(mutation)
    cna = torch.FloatTensor(cna)
    test_y = torch.FloatTensor(response.astype(int))
    model.eval()
    with torch.no_grad():
        expression_logit, mutation_logit, cna_logit = model.forward(
            expression, mutation, cna
        )
    X = np.stack([expression_logit, mutation_logit, cna_logit], axis=-1)
    X = np.nan_to_num(X)
    final_probabilities = logistic_regression.predict_proba(X)[:, 1]

    auc_validate = roc_auc_score(test_y, final_probabilities)
    auprc_validate = average_precision_score(test_y, final_probabilities)
    return auc_validate, auprc_validate
