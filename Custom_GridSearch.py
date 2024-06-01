import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from Classification.class_functions import train, model_test_accuracy
from Classification.conv_net_model import convnext_tiny, convnext_large
from Classification.GridMask import Grid

import csv

def grid_search(params, run, train_loader, val_loader, test_loader, ds_pos, ds_neg, log_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops = params['num_ops'], magnitude = params['magnitude']),
        Grid(True,True, rotate = 1, #never rotate image
            offset = params['offset'], ratio = params['ratio'], mode = params['mode'], prob = params['prob'])
        ])
    val_transform = transforms.Compose([])
    convnext_net = convnext_large(pretrained=False, in_22k=False, transform_train=train_transform, transform_val=val_transform, num_classes=2, drop_path_rate = params['drop_rate'])
    convnext_net.to(device)

    if params['apply_class_weights'] == True:
        # Calculate class weights
        num_class_1_samples = len(ds_pos)
        num_class_0_samples = len(ds_neg)
        total_samples = num_class_1_samples + num_class_0_samples
        class_weights = torch.tensor([total_samples / num_class_0_samples, total_samples / num_class_1_samples], dtype=torch.float).to(device)
    else:
        class_weights = None

    optimizer = optim.AdamW(convnext_net.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params['num_epochs'] - params['warmup_epochs'], eta_min=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    #Train
    convnext_net, train_losses, val_losses, val_accuracies = train(
        net = convnext_net, optimizer = optimizer, scheduler = scheduler, criterion = criterion,
        loader_train = train_loader, loader_val = val_loader,
        max_epochs = params['num_epochs'], early_stop = params['early_stop'], save_path = params['model_save_path'],
        run = f'{params["run"]}/{run}', save_freq = params['model_save_freq_epochs'])

    min_index = val_losses.index(min(val_losses))
    print(f'Best model at epoch: {min_index+1}')
    print(f'Train_loss: {train_losses[min_index]}')
    print(f'Val_loss = {val_losses[min_index]}')
    print(f'Val_accuracy = {val_accuracies[min_index]}')
    header = ["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "Params"]
    with open(log_path+'/Grid_logs.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
    
        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writerow(header)
    
        # Write the current iteration's information
        row = [
            min_index + 1,
            train_losses[min_index],
            val_losses[min_index],
            val_accuracies[min_index],
            params
            ]
        writer.writerow(row)

    return convnext_net, train_losses, val_losses, val_accuracies

    