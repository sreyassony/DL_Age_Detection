#!/usr/bin/env python
# coding: utf-8

# ## Pre processing

# In[ ]:


get_ipython().system('pip install ktrain')


# In[ ]:


# mounting google drive to colab virtual machine

from google.colab import drive
drive.mount('/content/drive', force_remount=True)


get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/DL_project/UTKFace_preprocessing/'")


# In[ ]:


import pandas as pd
import numpy as np
import os
import cv2
import h5py
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Activation, Dropout,
Flatten, Dense, Input, BatchNormalization, AveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


# loading the .h5 file into the notebook
# This file contains images in tensor/vector form
# we are loading it from the google drive

with h5py.File("UTK_predictors_X_64_pixels.h5", 'r') as hf:
  X = hf['images_in_vector'][:]


# In[ ]:


#listing the names of images in the image directory
names_and_age_ds = pd.read_csv('filenames_age.csv')


# In[ ]:


age = np.array(names_and_age_ds['age'])


# In[ ]:


# test train split 
X_train, X_dummy, y_train, y_dummy = train_test_split(X, age, 
                                                      test_size=0.3, 
                                                      shuffle=True, 
                                                      random_state=42)


X_valid, X_test, y_valid, y_test = train_test_split(X_dummy, y_dummy,
                                                    test_size=0.3,
                                                    shuffle=True, 
                                                    random_state=42)


# In[ ]:


print(f"Shape of the test data :\n{X_test.shape}, {y_test.shape}\n")
print(f"Percentage of the total dataset in the training data:\n {X_test.shape[0]/X.shape[0]}\n")


# ## Model building and Training

# In[ ]:


import ktrain
from ktrain import vision as vis
import re


# In[ ]:





# In[ ]:


#train test split
(train_data,test_data,preprocess)=vis.images_from_array(x_train=X_train,
                                                        y_train=y_train,
                                                        is_regression=True,
                                                        validation_data=(X_valid,y_valid),
                                                        random_state=1234,
                                                        data_aug=ImageDataGenerator(
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1, 
                                                                  horizontal_flip=True,
                                                                  rotation_range=45)
                                                        )


# In[ ]:


preprocess


# In[ ]:


#building a model
model=vis.image_regression_model('pretrained_resnet50',
                                 train_data=train_data,
                                 val_data=test_data)


# In[ ]:


model.summary()


# In[ ]:


#learner parameters
training=ktrain.get_learner(model=model,
                            train_data=train_data,
                            val_data=test_data,
                            batch_size=32)


# In[ ]:


training.lr_find(show_plot=True, restore_weights_only=True)


# In[ ]:


#training the model with 2 epochs and learning rate = 10^-2
history=training.fit_onecycle(1e-4,2)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(history.history['mae'], label='training loss')
plt.plot(history.history['val_mae'], label='validation loss')
plt.legend()
plt.show()


# In[ ]:


training.plot('loss')


# In[ ]:


training.view_top_losses(n=3)


# In[ ]:


#loading the previous stored weights
model.load_weights('/content/drive/MyDrive/DL_project/weights-02.hdf5')


# In[ ]:


history.history['val_mae']


# In[ ]:


#freezing first 15 layers and training the remaining layers
training.freeze(15)

history=training.fit_onecycle(1e-4,30,checkpoint_folder='/content/drive/MyDrive/DL_project/UTKFace_preprocessing/32/')


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(history.history['mae'], label='training loss')
plt.plot(history.history['val_mae'], label='validation loss')
plt.legend()
plt.show()


# In[ ]:


training.plot('loss')


# In[ ]:


#freezing first 15 layers and training the remaining layers
training.freeze(15)

training.fit_onecycle(1e-2,20,checkpoint_folder='/content/drive/MyDrive/DL_project/UTKFace_preprocessing/32/')


# In[ ]:


training.plot('loss')


# In[ ]:


training.plot('loss')


# In[ ]:


training.plot('loss')


# In[ ]:


train_mae=np.array([14.1385,17.4907 ,14.4631,10.5257 ,9.4719,8.7980, 8.6504 ,8.3766,8.1336,8.0925,8.0289, 8.0170 ,8.0214 ,7.8879,
7.8143,7.8784,7.7614,7.6938,7.4486,7.2618,7.0852,7.0043,6.8977,6.6906, 6.6439 ,6.4892 ,6.3133,6.2376, 
6.0027, 5.9075,5.7195,5.6195,5.6685,5.8706, 6.0306,6.1170,6.3331 , 6.5287 , 6.6095 ,6.6310 , 6.7126 ,
6.6725 ,6.7434,6.5385 ,6.2088 ,6.0740,6.0119,5.8323,5.6002 , 5.5435 ,5.4050 , 5.2935])


# In[ ]:


len(valid_mae)


# In[ ]:


valid_mae=np.array([16.5953,11.4460,13.0969,8.2018,13.5977,8.8735,9.6815,8.1986,10.2136,7.0512,7.1000,8.0563,16.3403,7.3043,
7.7062,11.2288,8.2474,7.5529,6.9048,9.7083,7.5295,7.7402,12.2855,6.7507,6.7756,5.9147,6.1738,5.8675,
5.7665,5.8833,6.2093,5.7254,7.0235,5.8266,5.9620,6.2198,7.1559,8.2050,7.3719,16.8459,8.1779,7.5440,7.8876,
6.2405,6.0302,5.7816,6.8135,6.5395,5.9477,5.7427,5.9327,5.6357])


# In[ ]:


x=np.linspace(0,51,52)


# In[ ]:





# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(x,train_mae,marker='*',label='train');
plt.plot(x,valid_mae,marker='*',label='validation');
plt.xlim(0,53)
plt.ylim(2,20)
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Model loss")
plt.legend(loc='best')
plt.show()


# ## Prediction

# In[ ]:


prediction=ktrain.get_predictor(training.model,preprocess)


# In[ ]:


preds=prediction.predict(X_test)


# In[ ]:


np.sum((np.squeeze(preds) - y_test)  <= 5)


# In[ ]:


def cum_score(max_tol, predictions, y_test):
  # total number of samples in the training set
  total_samples = len(y_test)
  # cumuliative score
  CS=[]
  for i in range(max_tol+1):
    abs_val = np.abs(np.squeeze(predictions) - y_test)
    # no of predictions that fall inside the tolerance level
    count = np.sum(abs_val <= i)
    CS.append(count/total_samples)
  return CS


# In[ ]:


cum_scores = cum_score(20, predictions=preds, y_test=y_test)


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(np.arange(21), cum_scores)
plt.xticks(np.arange(21))
plt.show()


# In[ ]:


df_cs = pd.Series(cum_scores)
df_cs.to_csv('cumuliative_score_resnet50.csv')


# In[ ]:


len(X_test)


# ## Reference

# https://www.youtube.com/watch?v=rwiPcSrPPQk&t=1490s

# https://github.com/amaiya/ktrain
