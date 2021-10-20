import os
import time
import numpy as np
import pandas as pd
import sys
import argparse

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from PIL import Image

from utils import get_args, print_header
from CACD_dataset import CACD_dataset
from CACD_dataloader import Transforms_Dataloader
from model import resnet18, niu_model, niu_forback
from train import train, evaluation_of_best_model, save_predictions
from train import compute_mae_and_mse


def main():

    args = get_args()
    IMG_DIR = args.img_dir
    TRAIN_CSV_PATH = args.train_csv_path
    VALID_CSV_PATH = args.valid_csv_path
    TEST_CSV_PATH = args.test_csv_path
    LOG_DIR = args.log_dir
    CUDA = args.cuda
    RAND_SEED = args.seed
    
    # checking whether device has cuda or not
    if CUDA >= 0:
        DEVICE = torch.device("cuda:%d" % CUDA)
    else:
        DEVICE = torch.device("cpu")
    
    # instantinating logging directory
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOGFILE = os.path.join(LOG_DIR, 'training.log')
    TEST_VALID_LOSS = os.path.join(LOG_DIR, 'training_validation.log')
    TEST_PREDICTIONS = os.path.join(LOG_DIR, 'test_predictions.log')

    # checking wheather random seed should be used
    RANDOM_SEED = RAND_SEED
    print(DEVICE)
    print_header(DEVICE, RANDOM_SEED, LOG_DIR, sys.argv[0],LOGFILE,
            TEST_VALID_LOSS)
    
    ####################################
    #            SETTINGS
    ####################################
    
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 250 

    # Architecture
    BATCH_SIZE = 256 
    
    # mean and standard deviation across three
    # channels of the whole training dataset
    mean = [0.5778, 0.4396, 0.3733]
    std = [0.2261, 0.1979, 0.1870]

    #######################################
    # Dataset and Dataloader
    #######################################

    data_trans_loader = Transforms_Dataloader(IMG_DIR,
                                              TRAIN_CSV_PATH,
                                              VALID_CSV_PATH,
                                              TEST_CSV_PATH,
                                              mean,std)
    
    
    
    # creating train, validation and test dataset
    train_loader, validation_loader, test_loader, len_train_dataset =\
            data_trans_loader.data_loaders()

       
    print(f'No of batches: {len_train_dataset//BATCH_SIZE}')
    #####################################
    #           TRAINING
    #####################################
    dtype = torch.FloatTensor
    torch.manual_seed(RANDOM_SEED)

    # model
#    model = resnet18()
    model = niu_model()
    model.type(dtype)

    # loss function: mean absolute error
    #loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    train(model, num_epochs, train_loader, validation_loader,
            test_loader, loss_fn, optimizer, DEVICE, len_train_dataset,
            BATCH_SIZE, LOGFILE, TEST_VALID_LOSS, LOG_DIR)
    
    # evaluating the best  model
    evaluation_of_best_model(model, train_loader, validation_loader, test_loader,
            LOGFILE, LOG_DIR, DEVICE)

    # saving predictions
    save_predictions(model, test_loader,TEST_PREDICTIONS,  DEVICE)



    
if __name__=="__main__":
    print("Starting training ...............")
    main()
    print("Finished")







    

