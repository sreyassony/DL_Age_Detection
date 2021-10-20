import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser(
            description = "This script is for extracting filenames of CACD\
            and labels into a csv file and then split into test and train\
            data")
    parser.add_argument("--img_dir", type=str, required=True,
        help="path to the image directory")
    parser.add_argument("--out_dir", type=str, required=True,
        help="path to where output should be stored")
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
    img_dir = args.img_dir
    out_dir = args.out_dir
    
    # extracting the names of the images from the directory
    names_list = np.array([img for img in tqdm(os.listdir(img_dir))\
            if img.endswith('.jpg')])
    # extracting the labels from the image names
    labels = np.array([img_name.split('_')[0] for img_name in tqdm(names_list)],
            dtype=np.float32)

    # Splitting the dataset int training, validation and testing set
    X_train, X_dummy, y_train, y_dummy = train_test_split(names_list,
            labels, test_size=0.2, random_state=42, shuffle=True,
            stratify=labels)
    X_valid, X_test, y_valid, y_test = train_test_split(X_dummy, y_dummy,
            test_size=0.05, random_state=42, shuffle=True, stratify=y_dummy)

    # dataframes for training, validation and test data set
    df_train = pd.DataFrame({'filename': X_train, 'age': y_train})
    df_valid = pd.DataFrame({'filename': X_valid, 'age': y_valid})
    df_test = pd.DataFrame({'filename': X_test, 'age': y_test})

    # saving the test train and validation datasets
    df_train.to_csv(os.path.join(out_dir, 'CACD_training.csv'), index=False)
    df_valid.to_csv(os.path.join(out_dir, 'CACD_valid.csv'), index=False)
    df_test.to_csv(os.path.join(out_dir, 'CACD_test.csv'), index=False)


if __name__=="__main__":
    main()




    

