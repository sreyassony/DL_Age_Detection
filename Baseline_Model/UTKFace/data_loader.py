import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utk_dataset import UTK_dataset 


# Tranforms and dataaugmentation
# Things to do:
# Random horizonatal flip and normalization of the data

class Transforms_Dataloader:
    """Apply transforms to the train, validation and test 
    dataset and give back the dataloader"""

    def __init__(self, img_dir, train_csv_path, valid_csv_path,
                 test_csv_path, mean, std, resize=64, rand_crop=60,
                 p_flip=0.5, n_workers=4, norm=False, random_flip=False,
                 imagenet=False, pin_memory=False, batch_size=256, shuffle=False):

        self.img_dir = img_dir
        self.train_csv_path = train_csv_path
        self.valid_csv_path = valid_csv_path
        self.test_csv_path = test_csv_path
        self.n_workers = n_workers
        self.resize = resize
        self.rand_crop = rand_crop
        self.p_flip = p_flip
        self.mean = mean 
        self.std = std
        self.norm = norm
        self.random_flip = random_flip
        self.imagenet = imagenet
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory 
        

    def custom_transform(self):
        
        if (self.norm and self.imagenet):
            train_transform =\
            transforms.Compose([transforms.Resize((self.resize, self.resize)),
                transforms.RandomCrop((self.rand_crop, self.rand_crop)),
                transforms.RandomHorizontalFlip(self.p_flip),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std= [0.229, 0.224, 0.225])])
        

        else:
            #train_transform =\
            #        transforms.Compose([transforms.Resize((self.resize, self.resize)),
            #            transforms.RandomCrop((self.rand_crop, self.rand_crop)),
            #            transforms.RandomHorizontalFlip(self.p_flip),
            #            transforms.ToTensor(),
            #            transforms.Normalize(self.mean, self.std)])
                        
            train_transform =\
                    transforms.Compose([transforms.Resize((self.resize, self.resize)),
                        transforms.RandomCrop((self.rand_crop, self.rand_crop)),
                        transforms.RandomHorizontalFlip(self.p_flip),
                        transforms.ToTensor()])
        
        
        validation_test_transform =\
                                    transforms.Compose([transforms.Resize((self.resize,
                                        self.resize)),
                                                        transforms.CenterCrop((self.rand_crop,
                                                            self.rand_crop)),
                                                        transforms.ToTensor()])

        return train_transform, validation_test_transform






    def data_loaders(self):
        """
        Adapted from https://github.com/a-martyn/resnet/blob/master/data_loader.py
        
        Utility function for loading and returning train and test
        multi-process iterators over the UTKFace dataset.
        If using CUDA, set pin_memory to True.
        
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - train_transform: pytorch transforms for the training set
        - test_transform: pytorch transofrms for the test set
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        
        Returns
        -------
        - train_loader: training set iterator.
        - validation_loader:  test set iterator.
        - test_loader: test set iterator
        """
        
        train_transform, validation_test_transform = self.custom_transform()
        # Load the datasets
        train_dataset = UTK_dataset(self.train_csv_path,
                                    self.img_dir,
                                    transform=train_transform)

        
        validation_dataset = UTK_dataset(self.valid_csv_path,
                                        self.img_dir,
                                        transform=validation_test_transform)

            
        test_dataset = UTK_dataset(self.test_csv_path,
                                self.img_dir,
                                transform=validation_test_transform)



        # Create loader objects
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.n_workers,
                                pin_memory=self.pin_memory)


        validation_loader = DataLoader(dataset=validation_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    num_workers=self.n_workers,
                                    pin_memory=self.pin_memory)

    
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.n_workers,
                                pin_memory=self.pin_memory)
            
        return train_loader, validation_loader, test_loader, len(train_dataset)




if __name__=="__main__":
    print("For data augmenation and creating dataloaders\n"
           "for train, validation and test datasets ")

