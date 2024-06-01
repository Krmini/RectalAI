import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tensorflow as tf
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import warnings

# Filter out UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def split_ds (ds, seed, train_split = 0.7, val_split = 0.2, orig_hight = 162, orig_width = 141, num_slices = 46):
    """
    Function to load a dataset, split it into train, validation and test
    """
    if seed != None:
        np.random.seed(seed)
    
    scans = list(np.array([x[1] for x,_ in ds]))
    scans = (np.reshape(scans,[len(ds), num_slices, orig_hight, orig_width, 1]))

    # Define the sizes of train, val, and test sets
    dataset_size = len(scans)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    # Define the indices for train, val, and test sets
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_scans = [scans[index] for index in train_indices]
    val_scans = [scans[index] for index in val_indices]
    test_scans = [scans[index] for index in test_indices]

    
    def split(scans):
        """
        Function to plit each scan into 2 scans of half the sequence length.
        Using tensorflow to match augmentation functions in augmentations for GAN.
        """

        wide_scans = []
        for scan in scans:
            for scan in [scan[::2], scan[1::2]]:
                with tf.device('/CPU:0'):
                    # scan = tf.image.per_image_standardization(scan)
                    scan = tf.image.resize(scan, [64, 64])
                    scan = tf.image.grayscale_to_rgb(scan)
                    scan = scan.numpy()
                scan = np.concatenate((scan), axis = 1)
                scan = torch.from_numpy(scan)
                scan = scan.permute(2, 0, 1)
                
                #normalize each wide scan to mean 0 and std 1
                mean = scan.mean()
                std = scan.std()
                scan = (scan - mean) / std

                wide_scans.append(scan)
        return wide_scans

    train_scans = split(train_scans)
    val_scans = split(val_scans)
    test_scans = split(test_scans)

    print(
        f'Dataset splits',
        f'\nTrain \t\t{train_split*100:g}% = {len(train_scans)}',
        f'\nValidation \t{val_split*100:g}% = {len(val_scans)}',
        f'\nTest \t\t{(1-train_split-val_split)*100:g}% = {len(test_scans)}')

    print('Image shape:', train_scans[0].shape,'\n')
    return train_scans, val_scans, test_scans


def concat_data(positive_scans, negative_scans, batch_size, workers,  shuffle=True):
    """
    Function to concatenate positive and negative datasets and add labels
    """
    data = np.concatenate([positive_scans, negative_scans], axis=0)
    targets = np.concatenate([np.ones(len(positive_scans)), np.zeros(len(negative_scans))], axis=0)
    dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(targets).long())
    tf.keras.backend.clear_session()
    torch.cuda.empty_cache()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = workers)
    return dataloader

def show_images(data_loader, num_images = 5):
    for batch in data_loader:
        input_data = batch[0]
        print('Labels:', batch[1][0:5])
        input_data = input_data[:num_images]
        grid_image = vutils.make_grid(input_data, nrow=1, padding=2, normalize=False)
        plt.figure(figsize=(20, num_images))
        plt.imshow(np.transpose(grid_image, (1, 2, 0)))
        plt.axis('off')
        plt.title('Navel     <<          Rectum          >>       Legs')
        plt.show()
        break


def train(net, optimizer, scheduler, criterion, loader_train, loader_val, max_epochs, early_stop, save_path, run, save_freq):
    print('Training: ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path+run, exist_ok=True)

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(max_epochs):
        # Train the model
        net.train()
        running_train_loss = 0.0
        for images, labels in loader_train:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            scheduler.step(epoch + len(loader_train) / len(loader_train))
            running_train_loss += train_loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(loader_train.dataset)
        train_losses.append(epoch_train_loss)

        # Validate the model
        net.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader_val:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                val_loss = criterion(outputs, labels).item()
                running_val_loss += val_loss * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(loader_val.dataset)
        val_losses.append(epoch_val_loss)

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        # Print statistics
        print(f"Epoch [{epoch+1}/{max_epochs}], "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Validation Loss: {epoch_val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.2f}%")
        
        if (epoch % save_freq == 0) or (epoch == max_epochs-1):
            torch.save(net.state_dict(), f"{save_path}{run}/Epoch_{epoch+1:03d}.zip")
        
        #apply early stopping
        if min(val_losses) < min(val_losses[-early_stop:]):
            print('Stopping early')
            break
    return net, train_losses, val_losses, val_accuracies



def plot_loss_acc(train_losses, val_losses, val_accuracies, save_dir = None):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(train_losses) + 1,1))

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    if save_dir:
        plt.savefig(f'{save_dir}/Loss_plot.png')
    plt.show()

    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    if save_dir:
        plt.savefig(f'{save_dir}/Accuracy_plot.png')
    plt.show()

def model_test_accuracy(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Print statistics
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
            f"Test Accuracy: {(100 * correct / total):.2f}%")

def model_test_metrics(model, test_loader, criterion, cutoff = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []
    sure_labels = []
    sure_outputs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs, 1)
            true_labels = labels
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
            
            all_labels.extend(true_labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            
            # Calculate absolute difference between logits
            differences = torch.abs(outputs[:, 0] - outputs[:, 1]).cpu().numpy()
            
            # Determine which samples are "sure"
            for i, diff in enumerate(differences):
                if diff >= cutoff:
                    sure_labels.append(true_labels[i].cpu().numpy())
                    sure_outputs.append(outputs[i].cpu().numpy())

    # Filter out unsure cases
    sure_labels = np.array(sure_labels)
    sure_outputs = np.array(sure_outputs)
    if sure_outputs.size > 0:
        sure_predicted = np.argmax(sure_outputs, axis=1)
        sure_probabilities = torch.softmax(torch.tensor(sure_outputs), dim=1)[:, 1].numpy()
    else:
        sure_predicted = np.array([])
        sure_probabilities = np.array([])

    # Compute accuracy
    accuracy = 100 * np.sum(sure_predicted == sure_labels) / len(sure_labels) if len(sure_labels) > 0 else 0

    # Compute AUC
    auc = roc_auc_score(sure_labels, sure_probabilities) if len(sure_labels) > 0 else 0

    # Compute Sensitivity and Specificity
    if len(sure_labels) > 0:
        tn, fp, fn, tp = confusion_matrix(sure_labels, sure_predicted).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        sensitivity = 0
        specificity = 0

    # Print statistics
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
          f"Test Accuracy: {accuracy:.2f}%, "
          f"Sensitivity (TPR): {sensitivity:.2f}, "
          f"Specificity (TNR): {specificity:.2f}, "
          f"AUC: {auc:.4f}, "
          f"Sure cases: {len(sure_labels)} , "
          f"Unsure cases: {len(all_labels) - len(sure_labels)}")

    # Plot ROC curve
    if len(sure_labels) > 0:
        fpr, tpr, _ = roc_curve(sure_labels, sure_probabilities)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return (test_loss / len(test_loader)), accuracy, sensitivity, specificity, auc, len(sure_labels), len(all_labels) - len(sure_labels)

def show_CAM(model, test_loader, num_maps = 3, layers = ['stages.2', 'stages.3'], alpha = 0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for case in [0, 1]:
        print('Class =', case)
        for layer in layers:
            print(layer)
            for images, labels in test_loader:
                maps = []
                image_n = 0
                for index in range(len(images)):
                    if labels[index] == case:
                        with SmoothGradCAMpp(model, target_layer = layer) as cam_extractor:
                            out = model(images[index].to(device).unsqueeze(0))
                            _, predicted_class = torch.max(out, 1)
                            # print(f"Correct prediction = {case == int(predicted_class)} \t\tClass prob.: [0: {out[0][0]:.3f}] [1: {out[0][1]:.3f}]")
                            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                        maps.append(activation_map)

                        grid_image = vutils.make_grid(images[index], nrow=1, padding=2, normalize=True)

                        result_original = overlay_mask(to_pil_image(grid_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=alpha)
                        plt.figure(figsize=(25, 5))
                        plt.imshow(result_original)
                        plt.axis('off')
                        plt.title(f'Labels: [True, Predicted] [{labels[index]}, {int(predicted_class)}]      -      Predicted class prob.: [0,1] [{out[0][0]:.3f},{out[0][1]:.3f}]')
                        plt.show()
                        image_n += 1
                    if image_n == num_maps:
                        break
                break
