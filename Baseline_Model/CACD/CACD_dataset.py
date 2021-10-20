import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CACD_dataset(Dataset):
    """Custom dataset for loading UTKFace images"""
    def __init__(self, csv_path, img_dir, transform=None):
        """
        csv_path: Load the csv file contains the filename and age
                  of the images.
        img_dir:  Path to the image directory
        """
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        # column in df with the image names
        self.img_paths = df.index.values
        self.age = df['age'].values
        self.transform = transform


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_paths[index]))

        if self.transform is not None:
            img = self.transform(img)
        label = self.age[index]

        return img, label

    def __len__(self):
        return self.age.shape[0]


if __name__=="__main__":
    print("Custom dataset for the CACD face dataset")
