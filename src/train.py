# ========================================
# DRUNET model training script
# ========================================

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from drunet import DRUNET
from dataloader import data_loader
import json
import os
import sys
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging
import time

import argparse

# Set up logging configuration
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to read a JSON file and return its data
def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data


if __name__ == '__main__':
    start_time = time.time()
    pmem = True if device == "cuda" else False  # set pin_memory to True

    parser = argparse.ArgumentParser(description='Training script.')

    parser.add_argument('--train_root', required=True, type=str, help='Root directory containing the training data.')
    parser.add_argument('--model_root', required=True, type=str, help='Root directory to store the trained model.')
    parser.add_argument('--model_name', required=True, type=str, help='Name of the trained model.')
    parser.add_argument('--config_file_path', type=str, default="src/config.json", help='Path to config file.')

    args = parser.parse_args()

    train_root = args.train_root

    model_root = os.path.join(args.model_root, args.model_name)
    os.makedirs(model_root, exist_ok=True)

    model_name = args.model_name

    writer = SummaryWriter(os.path.join(model_root, "tensorboard_logs"))  # Initialize writer for tensorboard log files

    config = read_json(args.config_file_path)  # Read the config file for model hyperparameters

    input_path = os.path.join(train_root, "input/")  # Input directory

    gt_path = os.path.join(train_root, "gt/")  # Ground truth directory

    validation_split_pc = config['validation_split_pc']

    train_dataloader, validation_dataloader = data_loader(input_path, gt_path, config['batch_size'], pmem, validation_split_pc)  # Initialize the dataloaders

    model = DRUNET(in_channels=1, out_channels=1, dropout=config['dropout'], dilations=[1, 2, 4, 8], features=[32, 64, 128])  # Initialize the model

    # Set up parallel training
    model = nn.DataParallel(model)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3])
    # summary(model)
    model.to(device)

    # Define the loss function, optimizer and learning rate scheduler

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-5, last_epoch=-1)

    checkpoint_save_path = os.path.join(model_root, "checkpoints")
    os.makedirs(checkpoint_save_path, exist_ok=True)

    model_save_path = os.path.join(model_root, "trained")
    os.makedirs(model_save_path, exist_ok=True)

    plot_save_path = os.path.join(model_root, "plots")
    os.makedirs(plot_save_path, exist_ok=True)

    hist = {"train_loss": [], "validation_loss": []}

    # Load best model if already exists, and continue training
    if os.path.exists(os.path.join(model_save_path, "saved_best_model_model.h5")):
         model_state = torch.load(os.path.join(model_save_path, "saved_best_model_model.h5"))
         model.load_state_dict(model_state['model_state_dict'])
         optimizer.load_state_dict(model_state['optimizer_state_dict'])
         scheduler.load_state_dict(model_state['lr_scheduler_state_dict'])


    min_validation_loss = 1e+12

    for e in tqdm(range(config['epochs'])):
        # Training
        model.train()

        total_train_loss = 0
        total_validation_loss = 0

        for (i, data) in enumerate(train_dataloader):
            optimizer.zero_grad()

            (x, y) = (data[0].cuda(), data[1].cuda())

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss

            # logging.info("Total Train loss: {:.6f}".format(total_train_loss))

        # Validation
        with torch.no_grad():
            model.eval()

            for (i, data) in enumerate(validation_dataloader):
                (x, y) = (data[0].to(device), data[1].to(device))
                pred = model(x)
                total_validation_loss += criterion(pred, y)

        mean_train_loss = total_train_loss / np.ceil(len(train_dataloader.dataset) / config['batch_size'])
        mean_validation_loss = total_validation_loss / np.ceil(len(validation_dataloader.dataset) / config['batch_size'])

        writer.add_scalars("loss", {'train': mean_train_loss, 'validation': mean_validation_loss}, e)

        # scheduler.step(mean_validation_loss)
        scheduler.step()

        hist["train_loss"].append(mean_train_loss.cpu().detach().numpy())
        hist["validation_loss"].append(mean_validation_loss.cpu().detach().numpy())

        logging.info("[INFO] EPOCH: {}/{}".format(e+1, config['epochs']))
        logging.info("Train loss: {:.6f}, Validation loss: {:.6f}".format(mean_train_loss, mean_validation_loss))

        # Store checkpoints
        if e % 10 == 0:
            torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': mean_train_loss,
                        'validation_loss': mean_validation_loss
                        },
                os.path.join(checkpoint_save_path, model_name + f'_checkpoint_{e}.h5'))

        # Update the best model based on validation loss
        if mean_validation_loss < min_validation_loss:
            min_validation_loss = mean_validation_loss
            torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': mean_train_loss,
                        'validation_loss': mean_validation_loss
                        },
                os.path.join(model_save_path, "saved_best_model_" + model_name + '.h5'))

    # Store the last model
    torch.save({'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict(),
                'train_loss': mean_train_loss,
                'validation_loss': mean_validation_loss
               },
               os.path.join(model_save_path, "saved_last_model_" + model_name + '.h5'))

    writer.close()

    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} sec.")

    # # Store the plot for losses
    # plt.figure()
    # plt.plot(hist["train_loss"], label="train_loss")
    # plt.plot(hist["validation_loss"], label="validation_loss")
    # plt.title("Loss History")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend(loc="best")
    # plt.savefig(os.path.join(plot_save_path, "loss_" + model_name + '.png'))
