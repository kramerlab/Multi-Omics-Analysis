import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import trange, tqdm
from models.pca_model import Classifier
from utils.network_training_util import (
    get_loss_fn,
    create_data_loader,
    create_sampler,
)
from scipy.stats import sem
from sklearn.decomposition import PCA

best_auroc = -1
cv_splits_inner = 5
sigmoid = torch.nn.Sigmoid()


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter(parameterization, x_e, x_m, x_c, y, device):
    variance_e = parameterization["variance_e"]
    variance_m = parameterization["variance_m"]
    variance_c = parameterization["variance_c"]
    dropout_rate = parameterization["dropout"]
    learning_rate = parameterization["learning_rate"]
    weight_decay = parameterization["weight_decay"]
    epochs = parameterization["epochs"]
    mini_batch = parameterization["mini_batch"]

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
        scaler_gdsc.fit(x_train_e)
        x_train_e = scaler_gdsc.transform(x_train_e)

        # Initialisation
        loss_fn = get_loss_fn(None, 0)

        pca_e = PCA(n_components=variance_e).fit(x_train_e)
        pca_m = PCA(n_components=variance_m).fit(x_train_m)
        pca_c = PCA(n_components=variance_c).fit(x_train_c)

        e_dimension = pca_e.n_components_
        m_dimension = pca_m.n_components_
        c_dimension = pca_c.n_components_

        input_size = e_dimension + m_dimension + c_dimension

        classifier_model = Classifier(input_size, dropout_rate).to(device)

        pca_optimiser = torch.optim.Adagrad(
            classifier_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        transformed_e = pca_e.transform(x_train_e)
        transformed_m = pca_m.transform(x_train_m)
        transformed_c = pca_c.transform(x_train_c)

        sampler = create_sampler(y_train)
        train_loader = create_data_loader(
            torch.FloatTensor(transformed_e),
            torch.FloatTensor(transformed_m),
            torch.FloatTensor(transformed_c),
            torch.FloatTensor(y_train),
            mini_batch,
            True,
            sampler,
        )

        for _ in trange(epochs, desc="Epoch"):
            train_pca(train_loader, classifier_model, pca_optimiser, loss_fn, device)

        # validate
        auc_validate, _ = test_pca(
            classifier_model,
            pca_e.transform(scaler_gdsc.transform(x_validate_e)),
            pca_m.transform(x_validate_m),
            pca_c.transform(x_validate_c),
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
    parameterization, x_train_e, x_train_m, x_train_c, y_train, device, pin_memory
):
    variance_e = parameterization["variance_e"]
    variance_m = parameterization["variance_m"]
    variance_c = parameterization["variance_c"]
    dropout_rate = parameterization["dropout"]
    learning_rate = parameterization["learning_rate"]
    weight_decay = parameterization["weight_decay"]
    epochs = parameterization["epochs"]
    mini_batch = parameterization["mini_batch"]

    train_scaler_gdsc = StandardScaler()
    train_scaler_gdsc.fit(x_train_e)
    x_train_e = train_scaler_gdsc.transform(x_train_e)

    pca_e = PCA(n_components=variance_e).fit(x_train_e)
    pca_m = PCA(n_components=variance_m).fit(x_train_m)
    pca_c = PCA(n_components=variance_c).fit(x_train_c)

    e_dimension = pca_e.n_components_
    m_dimension = pca_m.n_components_
    c_dimension = pca_c.n_components_

    transformed_e = pca_e.transform(x_train_e)
    transformed_m = pca_m.transform(x_train_m)
    transformed_c = pca_c.transform(x_train_c)

    loss_fn = get_loss_fn(0, 0)
    input_sizes = e_dimension + m_dimension + c_dimension
    pca_model = Classifier(input_sizes, dropout_rate).to(device)

    pca_optimiser = torch.optim.Adagrad(
        pca_model.parameters(),
        lr=learning_rate,
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
        torch.FloatTensor(transformed_e),
        torch.FloatTensor(transformed_m),
        torch.FloatTensor(transformed_c),
        torch.FloatTensor(y_train),
        mini_batch,
        pin_memory,
        sampler,
    )

    for _ in range(epochs):
        train_pca(train_loader, pca_model, pca_optimiser, loss_fn, device)
    return pca_model, train_scaler_gdsc, pca_e, pca_m, pca_c


def train_pca(train_loader, model, optimiser, loss_fn, device):
    y_true = []
    predictions = []
    model.train()
    for (data_e, data_m, data_c, target) in train_loader:
        if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
            optimiser.zero_grad()
            y_true.extend(target)

            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)

            input = torch.concat([data_e, data_m, data_c], axis=1)

            prediction = model.forward(input)
            loss = loss_fn(torch.squeeze(prediction), target)
            prediction = sigmoid(prediction)

            predictions.extend(prediction.cpu().detach())
            loss.backward()
            optimiser.step()
    y_true = torch.FloatTensor(y_true)
    predictions = torch.FloatTensor(predictions)
    auroc = roc_auc_score(y_true, predictions)
    return auroc


def test_pca(
    model,
    x_test_e,
    x_test_m,
    x_test_c,
    test_y,
    device,
):
    x_test_e = torch.FloatTensor(x_test_e).to(device)
    x_test_m = torch.FloatTensor(x_test_m).to(device)
    x_test_c = torch.FloatTensor(x_test_c).to(device)
    test_y = torch.FloatTensor(test_y.astype(int))
    model.eval()

    input = torch.concat([x_test_e, x_test_m, x_test_c], axis=1)
    predictions = model.forward(input)
    probabilities = sigmoid(predictions)
    auc_validate = roc_auc_score(test_y, probabilities.cpu().detach().numpy())
    auprc_validate = average_precision_score(
        test_y, probabilities.cpu().detach().numpy()
    )
    return auc_validate, auprc_validate
