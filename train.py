import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Unet_Multi_task_model import unet_1d_model
from EKGDataset import EKGDataset
from WSA import WSA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = unet_1d_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_and_validate(model, optimizer, device, validation_split=0.1):
    train_data = '../train_signal.npy'  # Ecg data set path
    train_label = '../train_label.npy'  # Patient label path
    train_label_12 = '../train_label_12.npy'  # Lead label path

    train_DS = EKGDataset(train_data, train_label, train_label_12)
    # Split the dataset into training and validation sets
    total_samples = len(train_DS)
    validation_size = int(validation_split * total_samples)
    training_size = total_samples - validation_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_DS, [training_size, validation_size])

    train_loader = DataLoader(train_dataset, 128, shuffle=True)
    val_loader = DataLoader(val_dataset, 32, shuffle=False)

    running_weighted_loss = 0.0
    lambda_reg = 0.001
    # Train
    model.train()
    for signals, labels, labels_12 in train_loader:
        signals = WSA(signals, 500, 4)
        signals = torch.tensor(signals, dtype=torch.float32)
        signals, labels, labels_12 = signals.to(device), labels.to(device), labels_12.to(device)
        optimizer.zero_grad()
        encoding, decoding, outputs_1, outputs_12 = model(signals)

        multi_class_loss = model.cal_loss_multi_class(outputs_1, labels)
        multi_label_loss = model.cal_loss_multi_label(((outputs_12 > 0.5).float()), labels_12)
        Unet_loss = model.cal_loss_Unet(decoding, signals)

        # Computes L2 regularization terms
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)  # Calculate the sum of squares of the parameters using the L2 norm

        # Adds the regularization term to the loss function
        multi_class_loss = multi_class_loss + lambda_reg * l2_reg
        multi_label_loss = multi_label_loss + lambda_reg * l2_reg
        Unet_loss = Unet_loss + lambda_reg * l2_reg


        weighted_loss = (0.4 * multi_class_loss) + (0.4 * multi_label_loss) + (0.2 * Unet_loss)

        weighted_loss.backward()
        optimizer.step()

        running_weighted_loss += weighted_loss.item()

    train_weighted_loss = running_weighted_loss / len(train_loader)

    print(f"train_weighted_loss: {train_weighted_loss:.4f}")


    # Validation
    model.eval()
    correct_1 = 0
    correct_12 = 0
    total = 0
    with torch.no_grad():
        for val_signals, val_labels, val_labels_12 in val_loader:
            val_signals = WSA(val_signals, 500, 4)
            val_signals = torch.tensor(val_signals, dtype=torch.float32)
            val_signals, val_labels, val_labels_12 = val_signals.to(device), val_labels.to(device), val_labels_12.to(device)
            encoding, decoding,outputs_1, outputs_12 = model(val_signals)

            _, predicted = torch.max(outputs_1.data, 1)
            correct_1 += (predicted == val_labels.squeeze()).sum().item()
            total += val_labels.size(0)

            # Calculate partially correct accuracy
            predicted_labels = (outputs_12 > 0.5).float()  # Use an appropriate threshold to determine the label's predictions
            correct_intersection = (predicted_labels == val_labels_12).sum(1)
            partial_accuracy = correct_intersection.float() / 12

            # Calculate the cumulative partial correct accuracy
            correct_12 += partial_accuracy.sum().item()

    val_accuracy = correct_1 / total
    accuracy_12 = correct_12 / total

    print(f"val_accuracy: {val_accuracy:.4f}")
    print(f"location_accuracy: {accuracy_12:.4f}")

    return train_weighted_loss,val_accuracy,accuracy_12





