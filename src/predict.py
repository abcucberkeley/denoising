# ========================================
# DRUNET model prediction script
# ========================================

import torch
import numpy as np
from drunet import DRUNET
import os
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchsummary import summary
import cpptiff

import json

import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")


# Function to read a JSON file and return its data
def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction script.')

    parser.add_argument('--predict_root', required=True, type=str, help='Root directory containing the test data.')
    parser.add_argument('--model_root', required=True, type=str, help='Root directory to store the trained model.')
    parser.add_argument('--model_name', required=True, type=str, help='Name of the trained model.')
    parser.add_argument('--config_file_path', type=str, default="src/config.json", help='Path to config file.')

    args = parser.parse_args()

    predict_root = args.predict_root

    input_path = os.path.join(predict_root, "input/")  # Root directory of input images

    # Extract filenames of the input images
    files = os.listdir(input_path)

    model_root = args.model_root
    model_name = args.model_name

    config = read_json(args.config_file_path)  # Read the config file for model hyperparameters

    # Load the DRUNET model
    model = DRUNET(in_channels=1, out_channels=1, dropout=config['dropout'], dilations=[1, 2, 4, 8], features=[32, 64, 128])
    model = nn.DataParallel(model)  # comment out if parallel training was not performed
    model_state = torch.load(os.path.join(model_root, model_name, "trained", "saved_best_model_" + model_name + ".h5"))
    model.load_state_dict(model_state['model_state_dict'])
    model.to(device)

    # Directory to store the predictions
    pred_path = os.path.join(predict_root, "predictions")
    os.makedirs(pred_path, exist_ok=True)

    with torch.no_grad():
        model.eval()

        for file in files:
            x = cpptiff.read_tiff(os.path.join(input_path, file))  # Read the input image
            # x -= 108  # Subtract offset if needed
            # x[x < 0] = 0  # Set negative values to 0
            # x = x / np.max(x)  # Normalize (based on whether the model has been trained on normalized data)

            x = torch.from_numpy(np.array([[x]])).to(device)  # move to GPU
            x = x.float()

            pred = model(x)  # Perform prediction
            pred = pred[0, 0].cpu().detach().numpy()  # move to CPU
            pred[pred < 0] = 0
            # restored = restored / np.max(restored)
            cpptiff.write_tiff(os.path.join(pred_path, file), pred)  # Save the prediction
