# ========================================
# Custom dataloader
# ========================================

import torch
import os
import numpy as np
import cpptiff
import time


# Custom Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_files, gt_files):
        self.input_files = input_files
        self.gt_files = gt_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x = cpptiff.read_tiff(self.input_files[idx])
        y = cpptiff.read_tiff(self.gt_files[idx])

        # # Adjust for offset if needed
        # x -= 108
        # x[x < 0] = 0
        #
        # # Normalize if needed
        # x = x/np.max(x)
        # y = y/np.max(y)
        #
        # x[x < 0] = 0
        # y[y < 0] = 0

        return np.array([x]), np.array([y])


# Custom dataloader function
def data_loader(input_path, gt_path, bsize, pmem, validation_split_pc=0.2):
    # Extract filenames of the input and ground truth data, and store in separate lists with one-to-one correspondence of input and ground truth filename
    gt_filenames = np.array(sorted(os.listdir(gt_path)))
    input_filenames = np.array(sorted(os.listdir(input_path)))

    num_files = len(gt_filenames)  # total number of files

    idx = np.arange(num_files)
    np.random.shuffle(idx)  # Randomly shuffle indices if needed

    nvalidation = int(num_files * validation_split_pc)
    ntrain = num_files - nvalidation

    # Extract the training data
    train_idx = idx[:ntrain]
    train_gt_filenames = gt_filenames[train_idx]
    train_input_filenames = input_filenames[train_idx]
    print('Training size: ', len(train_gt_filenames))

    train_gt_files = []
    train_input_files = []
    for i in range(ntrain):
        train_gt_files.append(os.path.join(gt_path, train_gt_filenames[i]))
        train_input_files.append(os.path.join(input_path, train_input_filenames[i]))

    train_data = Dataset(train_input_files, train_gt_files)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True, pin_memory=pmem)

    # Extract the validation data is the validation split percentage is greater than 0, else perform validation on the training data
    if validation_split_pc > 0:
        validation_idx = idx[ntrain:]
        validation_gt_filenames = gt_filenames[validation_idx]
        validation_input_filenames = input_filenames[validation_idx]
        print('Validation size: ', len(validation_gt_filenames))

        validation_gt_files = []
        validation_input_files = []
        for i in range(nvalidation):
            validation_gt_files.append(os.path.join(gt_path, validation_gt_filenames[i]))
            validation_input_files.append(os.path.join(input_path, validation_input_filenames[i]))

    else:
        validation_gt_files = train_gt_files.copy()
        print('Validation size: ', len(validation_gt_files))
        validation_input_files = train_input_files.copy()

    validation_data = Dataset(validation_input_files, validation_gt_files)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=bsize, pin_memory=pmem)

    return train_dataloader, validation_dataloader


if __name__ == '__main__':
    pass
