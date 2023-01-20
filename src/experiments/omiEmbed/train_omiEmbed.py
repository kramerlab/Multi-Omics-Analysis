import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from models.omiEmbed_model import VaeClassifierModel
from siamese_triplet.utils import AllTripletSelector
from utils.network_training_util import create_sampler, create_data_loader
from scipy.stats import sem

best_auroc = -1
cv_splits_inner = 5
lossFuncRecon = torch.nn.BCEWithLogitsLoss()
classifier_loss = torch.nn.BCEWithLogitsLoss()


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter(parameterization, x_e, x_m, x_c, y, device, pin_memory):
    torch.multiprocessing.set_sharing_strategy("file_system")
    mini_batch = parameterization["mini_batch"]
    lr_vae = parameterization["lr_vae"]
    lr_classifier = parameterization["lr_classifier"]
    weight_decay = parameterization["weight_decay"]
    epochs_phase = parameterization["epochs_phase"]
    latent_space_dim = parameterization["latent_space_dim"]
    dropout = parameterization["dropout"]
    k_kl = parameterization["k_kl"]
    k_embed = parameterization["k_embed"]
    dim_1B = parameterization["dim_1B"]
    dim_1A = parameterization["dim_1A"]
    dim_1C = parameterization["dim_1C"]
    class_dim_1 = parameterization["class_dim_1"]
    leaky_slope = parameterization["leaky_slope"]
    gamma = parameterization["gamma"]
    margin = parameterization["margin"]

    epochs_phase = int(epochs_phase / 3) if int(epochs_phase / 3) > 0 else 1

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

        e_in = x_train_e.shape[-1]
        m_in = x_train_m.shape[-1]
        c_in = x_train_c.shape[-1]
        omic_dims = (e_in, m_in, c_in)

        omi_embed_model = VaeClassifierModel(
            omic_dims,
            dropout,
            latent_space_dim,
            dim_1B,
            dim_1A,
            dim_1C,
            class_dim_1,
            leaky_slope,
        ).to(device)

        optimiser_embedding = torch.optim.Adagrad(
            params=omi_embed_model.netEmbed.parameters(),
            lr=lr_vae,
            weight_decay=weight_decay,
        )

        optimiser_classifier = torch.optim.Adagrad(
            params=omi_embed_model.netDown.parameters(),
            lr=lr_classifier,
            weight_decay=weight_decay,
        )

        train_omi_embed(
            train_loader,
            omi_embed_model,
            optimiser_embedding,
            optimiser_classifier,
            device,
            epochs_phase,
            k_kl,
            k_embed,
            gamma,
            margin,
        )

        # validate
        auc_validate, _ = test_omi_embed(
            omi_embed_model,
            scaler_gdsc,
            torch.FloatTensor(x_validate_e),
            torch.FloatTensor(x_validate_m),
            torch.FloatTensor(x_validate_c),
            y_validate,
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
    lr_vae = parameterization["lr_vae"]
    lr_classifier = parameterization["lr_classifier"]
    weight_decay = parameterization["weight_decay"]
    epochs_phase = parameterization["epochs_phase"]
    latent_space_dim = parameterization["latent_space_dim"]
    dropout = parameterization["dropout"]
    k_kl = parameterization["k_kl"]
    k_embed = parameterization["k_embed"]
    dim_1B = parameterization["dim_1B"]
    dim_1A = parameterization["dim_1A"]
    dim_1C = parameterization["dim_1C"]
    class_dim_1 = parameterization["class_dim_1"]
    leaky_slope = parameterization["leaky_slope"]
    gamma = parameterization["gamma"]
    margin = parameterization["margin"]

    epochs_phase = int(epochs_phase / 3) if int(epochs_phase / 3) > 0 else 1

    train_scaler_gdsc = StandardScaler()
    train_scaler_gdsc.fit(x_train_e)
    x_train_e = train_scaler_gdsc.transform(x_train_e)

    omic_dims = (x_train_e.shape[-1], x_train_m.shape[-1], x_train_c.shape[-1])
    omi_embed_model = VaeClassifierModel(
        omic_dims,
        dropout,
        latent_space_dim,
        dim_1B,
        dim_1A,
        dim_1C,
        class_dim_1,
        leaky_slope,
    ).to(device)

    optimiser_embedding = torch.optim.Adagrad(
        params=omi_embed_model.netEmbed.parameters(),
        lr=lr_vae,
        weight_decay=weight_decay,
    )

    optimiser_classifier = torch.optim.Adagrad(
        params=omi_embed_model.netDown.parameters(),
        lr=lr_classifier,
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

    train_omi_embed(
        train_loader,
        omi_embed_model,
        optimiser_embedding,
        optimiser_classifier,
        device,
        epochs_phase,
        k_kl,
        k_embed,
        gamma,
        margin,
    )
    return omi_embed_model, train_scaler_gdsc


def train_omi_embed(
    train_loader,
    model,
    optimiser_embedding,
    optimiser_classifier,
    device,
    epochs,
    k_kl,
    k_embed,
    gamma,
    margin,
):

    if gamma > 0:
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)
        triplet_selector = AllTripletSelector()
    else:
        triplet_loss_fn = None
        triplet_selector = None

    y_true = []
    for _ in range(epochs):
        model.netEmbed.train()
        model.netDown.eval()
        for (data_e, data_m, data_c, target) in train_loader:
            if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
                if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
                    optimiser_embedding.zero_grad()
                    y_true.extend(target)
                    data_e = data_e.to(device)
                    data_m = data_m.to(device)
                    data_c = data_c.to(device)
                    _, recon_x, mean, log_var = model.encode(data_e, data_m, data_c)
                    loss_kl = kl_loss(mean, log_var)
                    reconstruction_loss = (
                        lossFuncRecon(recon_x[0], data_e)
                        + lossFuncRecon(recon_x[1], data_m)
                        + lossFuncRecon(recon_x[2], data_c)
                    )
                    loss = k_kl * loss_kl + reconstruction_loss

                    if gamma > 0:
                        triplets = triplet_selector.get_triplets(mean, target)
                        triplet_loss = triplet_loss_fn(
                            mean[triplets[:, 0], :],
                            mean[triplets[:, 1], :],
                            mean[triplets[:, 2], :],
                        )
                        loss += triplet_loss
                    loss.backward()
                    optimiser_embedding.step()

    for _ in range(epochs):
        model.netEmbed.eval()
        model.netDown.train()
        for (data_e, data_m, data_c, target) in train_loader:
            if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
                if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
                    optimiser_classifier.zero_grad()
                    y_true.extend(target)
                    data_e = data_e.to(device)
                    data_m = data_m.to(device)
                    data_c = data_c.to(device)
                    target = target.to(device)
                    logit = model.classify(data_e, data_m, data_c)
                    loss = classifier_loss(target, torch.squeeze(logit))
                    loss.backward()
                    optimiser_classifier.step()

    for _ in range(epochs):
        model.netEmbed.train()
        model.netDown.train()
        for (data_e, data_m, data_c, target) in train_loader:
            if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
                optimiser_embedding.zero_grad()
                optimiser_classifier.zero_grad()
                y_true.extend(target)
                data_e = data_e.to(device)
                data_m = data_m.to(device)
                data_c = data_c.to(device)
                target = target.to(device)
                z, recon_x, mean, log_var, logit = model.encode_and_classify(
                    data_e, data_m, data_c
                )
                loss_kl = kl_loss(mean, log_var)
                classification_loss = classifier_loss(target, torch.squeeze(logit))
                reconstruction_loss = (
                    lossFuncRecon(recon_x[0], data_e)
                    + lossFuncRecon(recon_x[1], data_m)
                    + lossFuncRecon(recon_x[2], data_c)
                )
                loss = (
                    k_embed * (k_kl * loss_kl + reconstruction_loss)
                    + classification_loss
                )
                loss.backward()
                optimiser_embedding.step()
                optimiser_classifier.step()


sigmoid = torch.nn.Sigmoid()


def test_omi_embed(model, scaler, extern_e, extern_m, extern_c, test_r):
    model = model.cpu()
    extern_e = torch.FloatTensor(scaler.transform(extern_e))
    extern_m = torch.FloatTensor(extern_m)
    extern_c = torch.FloatTensor(extern_c)

    test_y = torch.FloatTensor(test_r.astype(int))
    model.eval()
    with torch.no_grad():
        logit = model.classify(extern_e, extern_m, extern_c)
    probabilities = sigmoid(logit)
    auc_validate = roc_auc_score(test_y, probabilities)
    auprc_validate = average_precision_score(test_y, probabilities)
    return auc_validate, auprc_validate


def kl_loss(mean, log_var, reduction="mean"):
    part_loss = 1 + log_var - mean.pow(2) - log_var.exp()
    if reduction == "mean":
        loss = -0.5 * torch.mean(part_loss)
    else:
        loss = -0.5 * torch.sum(part_loss)
    return loss
