import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np

from EKGDataset import EKGDataset
from WSA import WSA
from Unet_Multi_task_model import unet_1d_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = unet_1d_model().to(device)

def test():
    model.load_state_dict(torch.load("./model_pt/Loss(4,4,2).pt"), strict=False)

    test_data = './test_data/test_signal.npy'  # Ecg data set path
    test_label_1 = './test_data/test_label.npy'  # Patient label path
    test_label_12 = './test_data/test_label_12.npy'  # Lead label path
    test_dataset = EKGDataset(test_data, test_label_1, test_label_12)
    test_loader = DataLoader(test_dataset, 64, shuffle=True)
    all_true_labels = []
    all_predicted_labels = []
    correct_12 = 0
    total = np.load('test_data/test_signal.npy').shape[0]
    y_true=list()
    y_pred=list()
    model.eval()
    with torch.no_grad():
        for signals, labels_1, labels_12 in test_loader:
            signals = WSA(signals, 500, 4)
            signals = torch.tensor(signals, dtype=torch.float32)
            signals, labels_1, labels_12 = signals.to(device), labels_1.to(device), labels_12.to(device)
            encoding, decoding,outputs_1, outputs_12 = model(signals)

            predicted_labels = (outputs_12 > 0.5).float()
            correct_intersection = (predicted_labels == labels_12).sum(1)
            partial_accuracy = correct_intersection.float() / 12


            correct_12 += partial_accuracy.sum().item()

            all_true_labels.append(labels_12.cpu().numpy())
            all_predicted_labels.append(predicted_labels.cpu().numpy())

            _, predicted = torch.max(outputs_1.data, 1)
            y_pred.append(predicted.detach().cpu().numpy())
            y_true.append(labels_1.squeeze().detach().cpu().numpy())

    accuracy_12 = correct_12 / total

    print(f"location_accuracy: {accuracy_12:.3f}")

    print(metrics.classification_report(np.concatenate(y_true), np.concatenate(y_pred),digits=3))


if __name__ == '__main__':
    test()
