import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(
                        description="This script is for preprocessing the data\
                                and splitting the dataset into training,\
                                validation and test dataset and saving them\
                                in csv files",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--img_dir", type=str, required=True,
            help="path to the image directory")
    parser.add_argument("--out_dir", type=str, required=True,
            help="path to the where output should be stored")
    parser.add_argument("--cutoff", type=int, default=1000,
                        help="maximum number of samples in each age class")
    parser.add_argument("--start_age", type=int, default=21,
                        help="Starting age for age sampling")
    parser.add_argument("--end_age", type=int, default=60,
                        help="end range of the age for sampling")
    args = parser.parse_args()
    return args


def balance_data(cut_off, ds, start_age, end_age):

    """This function is to balance the data if we need it.
    Lot of ages in the UTK dataset are under 50. if cut off is given
    for example 70, this function will limit the number of samples
    which is more than 70, to 70 and keep the samples, which are
    less than 70 as it is

    ds: array with image names and ages
    out
    """
    rng = np.random.default_rng(seed=0)


    ds_out = []
    for i in range(start_age, end_age+1):
        # masking all the entries in the column if it matches the age
        mask = ds[:, 1] == str(i)
        # checking number of occurences of the ages greater than cutoff
        if np.sum(mask) > cut_off:
            masked = ds[mask]
            rng.shuffle(masked)
            ds_out.append(masked[:cut_off])
        else:
            ds_out.append(ds[mask])

    ds_out = np.array(ds_out, dtype=object)
    #stacking the list vertically to reduce array dimension
    return np.vstack(ds_out)




def main():

    args = get_args()
    root_dir = args.img_dir
    out_dir = args.out_dir
    start_age = args.start_age
    end_age = args.end_age
    cut_off = args.cutoff


    # extracting the names of the images from the directory
    names_list = [img for img in os.listdir(root_dir) if img.endswith('.jpg')]


    #extracting the labels from the image names
    labels = [img_name.split('_')[0] for img_name in names_list]


    #dataset
    ds = np.array([names_list, labels]).T

    dataset = pd.DataFrame(balance_data(cut_off, ds, start_age, end_age), 
            columns=['filename', 'age'])
    
    dataset.age = dataset.age.astype(np.float32)


    # Splitting test dataset, validation_dataset and training set
    X_train, X_dummy, y_train, y_dummy = train_test_split(dataset['filename'],
                                                          dataset['age'],
                                                          test_size=0.2,
                                                          random_state=42,
                                                          shuffle=True,
                                                          stratify=dataset['age'])


    X_valid, X_test, y_valid, y_test = train_test_split(X_dummy,
                                                        y_dummy,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=y_dummy)


    # dataframes for training, validation and test data set
    
    df_train = pd.DataFrame({'filename': X_train, 'age': y_train})
    df_valid = pd.DataFrame({'filename': X_valid, 'age': y_valid})
    df_test = pd.DataFrame({'filename': X_test, 'age': y_test})
    

    # saving test train and validation datasets
    df_train.to_csv(os.path.join(out_dir, f'training_set_{start_age}_{end_age}.csv'), index=False)
    df_valid.to_csv(os.path.join(out_dir, f'validation_set_{start_age}_{end_age}.csv'), index=False)
    df_test.to_csv(os.path.join(out_dir, f'test_set_{start_age}_{end_age}.csv'), index=False)


if __name__=="__main__":
    main()












