import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
from models.early_integration_model import EarlyIntegration
from utils.network_training_util import get_loss_fn, create_sampler
from scipy.stats import sem

best_auroc = -1
cv_splits_inner = 5
sigmoid = torch.nn.Sigmoid()


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter(parameterization, x, y, device, pin_memory):
    mini_batch = parameterization['mini_batch']
    h_dim = parameterization['h_dim']
    lr = parameterization['lr']
    dropout_rate = parameterization['dropout_rate']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    aucs_validate = []
    iteration = 1
    skf = StratifiedKFold(n_splits=cv_splits_inner)
    for train_index, validate_index in tqdm(skf.split(x, y), total=skf.get_n_splits(),
                                            desc="k-fold"):
        x_train_e = x[train_index]
        y_train = y[train_index]

        x_validate_e = x[validate_index]
        y_validate = y[validate_index]

        scaler_gdsc = StandardScaler()
        x_train_e = scaler_gdsc.fit_transform(x_train_e)

        # Initialisation
        sampler = create_sampler(y_train)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                                   num_workers=8, pin_memory=pin_memory, drop_last=True,
                                                   sampler=sampler)

        _, ie_dim = x_train_e.shape

        loss_fn = get_loss_fn(margin, gamma)

        early_integration_model = EarlyIntegration(ie_dim, h_dim, dropout_rate, ).to(device)

        moli_optimiser = torch.optim.Adagrad(early_integration_model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in trange(epochs, desc='Epoch'):
            train_early_integration(train_loader, early_integration_model, moli_optimiser, loss_fn, device, gamma)

        # validate
        auc_validate, _ = test_early_integration(early_integration_model, scaler_gdsc, x_validate_e, y_validate, device)
        aucs_validate.append(auc_validate)

        if iteration < cv_splits_inner:
            open_folds = cv_splits_inner - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(np.concatenate([aucs_validate, remaining_best_results]))
            if check_best_auroc(best_possible_mean):
                print('Skip remaining folds.')
                break
        iteration += 1

    mean = np.mean(aucs_validate)
    set_best_auroc(mean)
    standard_error_of_mean = sem(aucs_validate)

    return {'auroc': (mean, standard_error_of_mean)}


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def set_best_auroc(new_auroc):
    global best_auroc
    if new_auroc > best_auroc:
        best_auroc = new_auroc


def train_final(parameterization, x_train_e, y_train, device, pin_memory):
    mini_batch = parameterization['mini_batch']
    h_dim = parameterization['h_dim']
    lr = parameterization['lr']
    dropout_rate = parameterization['dropout_rate']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    train_scaler_gdsc = StandardScaler()
    train_scaler_gdsc.fit(x_train_e)
    x_train_e = train_scaler_gdsc.transform(x_train_e)

    ie_dim = x_train_e.shape[-1]

    loss_fn = get_loss_fn(margin, gamma)

    early_integration_model = EarlyIntegration(ie_dim, h_dim, dropout_rate).to(device)

    optimiser = torch.optim.Adagrad(early_integration_model.parameters(), lr=lr, weight_decay=weight_decay)

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                               num_workers=8, pin_memory=pin_memory, drop_last=True,
                                               sampler=sampler)

    for _ in range(epochs):
        train_early_integration(train_loader, early_integration_model, optimiser, loss_fn, device, gamma)
    return early_integration_model, train_scaler_gdsc


def train_early_integration(train_loader, model, optimiser, loss_fn, device, gamma):
    y_true = []
    model.train()
    for (data, target) in train_loader:
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            optimiser.zero_grad()
            y_true.extend(target)

            data = data.to(device)
            target = target.to(device)
            prediction = model.forward_with_features(data)
            if gamma > 0:
                loss = loss_fn(prediction, target)
            else:
                loss = loss_fn(prediction[0], target)
            loss.backward()
            optimiser.step()


def test_early_integration(model, scaler, extern_concat, test_r, device):
    x_test_e = torch.FloatTensor(scaler.transform(extern_concat)).to(device)
    test_y = torch.FloatTensor(test_r.astype(int))
    model.eval()
    predictions = model.forward_with_features(x_test_e)
    probabilities = sigmoid(predictions[0])
    auc_validate = roc_auc_score(test_y, probabilities.cpu().detach().numpy())
    auprc_validate = average_precision_score(test_y, probabilities.cpu().detach().numpy())
    return auc_validate, auprc_validate
