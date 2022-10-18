import numpy as np
import torch
import torch.utils.data
import torch.nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from tqdm import trange

from siamese_triplet.utils import AllTripletSelector

sigmoid = torch.nn.Sigmoid()


def train(train_loader, model, optimiser, loss_fn, device, gamma):
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

            prediction = model.forward(data_e, data_m, data_c)
            if gamma > 0:
                loss = loss_fn(prediction, target)
            else:
                loss = loss_fn(torch.squeeze(prediction[0]), target)
            prediction = sigmoid(prediction[0])

            predictions.extend(prediction.cpu().detach())
            loss.backward()
            optimiser.step()
    y_true = torch.FloatTensor(y_true)
    predictions = torch.FloatTensor(predictions)
    auroc = roc_auc_score(y_true, predictions)
    return auroc


class BceWithTripletsToss:
    def __init__(self, gamma, triplet_selector, trip_criterion):
        self.gamma = gamma
        self.trip_criterion = trip_criterion
        self.triplet_selector = triplet_selector
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        super(BceWithTripletsToss, self).__init__()

    def __call__(self, predictions, target):
        prediction = torch.squeeze(predictions[0])
        zt = predictions[1]
        triplets = self.triplet_selector.get_triplets(zt, target)
        target = torch.squeeze(target.view(-1, 1))
        loss = self.gamma * self.trip_criterion(
            zt[triplets[:, 0], :], zt[triplets[:, 1], :], zt[triplets[:, 2], :]
        ) + self.bce_with_logits(prediction, target)
        return loss


def read_and_transpose_csv(path):
    csv_data = pd.read_csv(path, sep="\t", index_col=0, decimal=",")
    return pd.DataFrame.transpose(csv_data)


def calculate_mean_and_std_auc(result_dict, result_file, drug_name):
    result_file.write(f"\tMean Result for {drug_name}:\n\n")
    for result_name, result_value in result_dict.items():
        mean = np.mean(result_value)
        std = np.std(result_value)
        max_value = np.max(result_value)
        min_value = np.min(result_value)
        result_file.write(f"\t\t{result_name} max: {max_value}\n")
        result_file.write(f"\t\t{result_name} min: {min_value}\n")
        result_file.write(f"\t\t{result_name} mean: {mean}\n")
        result_file.write(f"\t\t{result_name} std: {std}\n")
        result_file.write("\n")


def test(moli_model, scaler, x_test_e, x_test_m, x_test_c, test_y, device):
    x_test_e = torch.FloatTensor(scaler.transform(x_test_e)).to(device)
    x_test_m = torch.FloatTensor(x_test_m).to(device)
    x_test_c = torch.FloatTensor(x_test_c).to(device)
    test_y = torch.FloatTensor(test_y.astype(int))
    moli_model.eval()
    predictions = moli_model.forward(x_test_e, x_test_m, x_test_c)
    probabilities = sigmoid(predictions[0])
    auc_validate = roc_auc_score(test_y, probabilities.cpu().detach().numpy())
    auprc_validate = average_precision_score(
        test_y, probabilities.cpu().detach().numpy()
    )
    return auc_validate, auprc_validate


def create_data_loader(
    x_test_e, x_test_m, x_test_c, test_y, train_batch_size, pin_memory, sampler=None
):
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test_e),
        torch.FloatTensor(x_test_m),
        torch.FloatTensor(x_test_c),
        torch.FloatTensor(test_y),
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=sampler,
    )
    return loader


def get_triplet_selector():
    return AllTripletSelector()


def get_loss_fn(margin, gamma, triplet_selector):
    if triplet_selector is not None and gamma > 0:
        trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
        return BceWithTripletsToss(gamma, triplet_selector, trip_criterion)
    else:
        return torch.nn.BCEWithLogitsLoss()


def feature_selection(gdsce, gdscm, gdscc):
    selector = VarianceThreshold(1)
    selector.fit_transform(gdsce)
    gdsce = gdsce[gdsce.columns[selector.get_support(indices=True)]]

    selector = VarianceThreshold(0.00015)
    selector.fit_transform(gdscm)
    gdscm = gdscm[gdscm.columns[selector.get_support(indices=True)]]

    selector = VarianceThreshold(0.2)
    selector.fit_transform(gdscc)
    gdscc = gdscc[gdscc.columns[selector.get_support(indices=True)]]

    return gdsce, gdscm, gdscc


def create_sampler(y_train):
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    return sampler


def train_encoder(
    epochs,
    optimizer,
    triplet_selector,
    device,
    encoder,
    train_loader,
    trip_loss_fun,
    omic_number,
):
    encoder.train()
    for _ in trange(epochs):
        for data in train_loader:
            single_omic_data = data[omic_number]
            target = data[-1]
            if torch.mean(target) != 0.0 and torch.mean(target) != 1.0:
                optimizer.zero_grad()
                single_omic_data = single_omic_data.to(device)

                encoded_data = encoder(single_omic_data)
                triplets = triplet_selector.get_triplets(encoded_data, target)
                loss = trip_loss_fun(
                    encoded_data[triplets[:, 0], :],
                    encoded_data[triplets[:, 1], :],
                    encoded_data[triplets[:, 2], :],
                )
                loss.backward()
                optimizer.step()
    encoder.eval()


def train_validate_classifier(
    classifier_epoch,
    device,
    e_supervised_encoder,
    m_supervised_encoder,
    c_supervised_encoder,
    train_loader,
    classifier_optimizer,
    x_val_e,
    x_val_m,
    x_val_c,
    y_val,
    classifier,
):
    train_classifier(
        classifier,
        classifier_epoch,
        train_loader,
        classifier_optimizer,
        e_supervised_encoder,
        m_supervised_encoder,
        c_supervised_encoder,
        device,
    )

    with torch.no_grad():
        classifier.eval()
        """
            inner validation
        """
        encoded_val_E = e_supervised_encoder(x_val_e)
        encoded_val_M = m_supervised_encoder(torch.FloatTensor(x_val_m).to(device))
        encoded_val_C = c_supervised_encoder(torch.FloatTensor(x_val_c).to(device))
        test_Pred = classifier(encoded_val_E, encoded_val_M, encoded_val_C)
        test_y_pred = test_Pred.cpu()
        test_y_pred = sigmoid(test_y_pred)
        val_auroc = roc_auc_score(y_val, test_y_pred.detach().numpy())

    return val_auroc


def train_classifier(
    classifier,
    classifier_epoch,
    train_loader,
    classifier_optimizer,
    e_supervised_encoder,
    m_supervised_encoder,
    c_supervised_encoder,
    device,
):
    bce_loss_function = torch.nn.BCEWithLogitsLoss()
    for _ in range(classifier_epoch):
        classifier.train()
        for _, (dataE, dataM, dataC, target) in enumerate(train_loader):
            classifier_optimizer.zero_grad()
            dataE = dataE.to(device)
            dataM = dataM.to(device)
            dataC = dataC.to(device)
            target = target.to(device)

            encoded_e = e_supervised_encoder(dataE)
            encoded_m = m_supervised_encoder(dataM)
            encoded_c = c_supervised_encoder(dataC)
            predictions = classifier(encoded_e, encoded_m, encoded_c)
            cl_loss = bce_loss_function(
                torch.squeeze(predictions), torch.squeeze(target)
            )
            cl_loss.backward()
            classifier_optimizer.step()
    classifier.eval()


def super_felt_test(
    x_test_e,
    x_test_m,
    x_test_c,
    y_test,
    device,
    final_c_supervised_encoder,
    final_classifier,
    final_e_supervised_encoder,
    final_m_supervised_encoder,
    final_scaler_gdsc,
):
    x_test_e = torch.FloatTensor(final_scaler_gdsc.transform(x_test_e))
    encoded_test_E = final_e_supervised_encoder(torch.FloatTensor(x_test_e).to(device))
    encoded_test_M = final_m_supervised_encoder(torch.FloatTensor(x_test_m).to(device))
    encoded_test_C = final_c_supervised_encoder(torch.FloatTensor(x_test_c).to(device))
    test_prediction = final_classifier(encoded_test_E, encoded_test_M, encoded_test_C)
    test_y_true = y_test
    test_y_prediction = test_prediction.cpu().detach().numpy()
    test_AUC = roc_auc_score(test_y_true, test_y_prediction)
    test_AUCPR = average_precision_score(test_y_true, test_y_prediction)
    return test_AUC, test_AUCPR
