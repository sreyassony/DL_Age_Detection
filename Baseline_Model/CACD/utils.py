import argparse
import pandas as pd
import numpy as np
import torch
import sys

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from CACD_dataset import CACD_dataset


def get_args():

    parser = argparse.ArgumentParser(
                        description="This script is for training the model\
                                Takes image directory, csvpaths for test valid\
                                and train_dataset as arguments",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--img_dir", type=str, required=True,
            help="path to the image directory")
    parser.add_argument("--log_dir", type=str, required=True,
            help="path to the where output should be stored")
    parser.add_argument("--train_csv_path", type=str, required=True,
            help="path to the training data(csv_file with image names and age")
    parser.add_argument("--valid_csv_path", type=str, required=True,
            help="path to validation data csv file")
    parser.add_argument("--test_csv_path", type=str, required=True,
            help="path to testing data csv file")
    parser.add_argument("--cuda", type=int, default=-1,
            help="cuda or not, if yes enter a number >= 0")
    parser.add_argument("--seed", type=int, default=42,
            help="random seed for reproducibility")
    args = parser.parse_args()
    return args



def print_header(DEVICE, RANDOM_SEED, LOG_DIR, sysarg, LOGFILE, TEST_VALID_LOSS):
    header = []
    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Output Path: %s' % LOG_DIR)
    header.append('Script: %s' % sys.argv[0])

    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()

    with open(TEST_VALID_LOSS, 'w') as f:
        f.write('Training loss , Validation loss')
        f.flush()


#train_path = 'xxxxxx'
#img_dir = 'xxxxx'
#dataset = UTK_dataset(train_path, img_dir,
#        transform=transforms.Compose([transforms.ToTensor()]))

# for calculating the mean and standard deviation of the training set

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in tqdm(dataloader):
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std



if __name__=="__main__":
    print("Utilities for training the model")
   # print(get_mean_and_std(dataset))
