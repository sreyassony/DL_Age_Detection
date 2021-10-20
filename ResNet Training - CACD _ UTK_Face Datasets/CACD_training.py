#!/usr/bin/env python
# coding: utf-8

# ## **Preprocessing**

# In[ ]:


# mounting google drive to colab virtual machine

from google.colab import drive
drive.mount('/content/drive', force_remount=True)


get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/DL_Project/CACD_preprocess/'")


# In[ ]:


get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/DL_Project/update'")


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('pip install ktrain')


# In[ ]:


get_ipython().system('pip install face_recognition')


# In[ ]:


import ktrain
from ktrain import vision as vis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import cv2
import os
import face_recognition
from PIL import Image
import shutil
import re
import glob
import ntpath
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import pickle


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system("unzip '/content/drive/My Drive/DL_Project/CACD_preprocess/CACD_centered.zip' -d /content/")


# In[ ]:


img_dir = '/content/CACD_centered/'


# In[ ]:


#listing the names of images in the image directory
list_of_img_names = os.listdir(img_dir)


# In[ ]:


#taking out the age from the image names for our target variable
age = np.array([names.split('_')[0] for names in list_of_img_names], 
               dtype=np.float32)


# In[ ]:



font = {'family': 'serif',
        'color':  '#008D08',
        'weight': 'bold',
        'size': 12,
        }
#plt.rcParams["figure.figsize"] = (22,10)
plt.figure(figsize=(18,6))
arr= plt.hist(x=age, bins=np.count_nonzero(np.unique(age)), align='mid',rwidth=.5,color='b')
for i in range(np.count_nonzero(np.unique(age))):
    plt.text(arr[1][i],arr[0][i]+100,int(arr[0][i]),fontdict=font,rotation=-90)
plt.xlabel('age')
plt.ylabel('Frequency')
plt.title('Distribution of age')
bottom, top = plt.ylim()
plt.ylim((bottom,top+500))


plt.show()


# ## Training the model

# In[ ]:


#Image pattern 
pattern=r'([\d]+)_\w+_\d+.jpg$'
p=re.compile(pattern)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


#train test split
(train_data,test_data,preprocess)=vis.images_from_fname(img_dir,
                                                        pattern=pattern,
                                                        val_pct=0.2,
                                                        is_regression=True,
                                                        random_state=42,
                                                        data_aug=ImageDataGenerator(
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1, 
                                                                  horizontal_flip=True,
                                                                  rotation_range=45)
)


# In[ ]:


#available image regression model
vis.print_image_regression_models()


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
                            batch_size=64)


# In[ ]:


training.lr_find(show_plot=True, restore_weights_only=True)


# In[ ]:


training.set_weight_decay(10**-3)


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


model.load_weights('/content/drive/My Drive/DL_Project/update/weights-03.hdf5')


# In[ ]:


#training the model with 2 epochs and learning rate = 10^-4
training.fit_onecycle(1e-4,2,checkpoint_folder='/content/drive/MyDrive/DL_project/')


# In[ ]:


print(training.get_weight_decay())


# In[ ]:


mae=[9.7028, 7.9876, 6.9388, 6.1731, 5.9146, 5.43145751953125, 5.4384589195251465, 5.487170219421387, 5.204601764678955, 4.83113431930542, 4.7636, 4.95, 5.0877, 4.7607, 4.8367, 4.9354, 4.7024, 4.365, 4.3475799560546875, 4.521537780761719, 4.697944164276123, 4.4693803787231445, 4.117332935333252, 4.131679534912109, 4.466404438018799, 4.104224681854248]
val_mae=[8.0117, 7.5809, 5.9674, 6.3078, 5.6124, 5.449876308441162, 5.407055854797363, 5.49222469329834, 5.0487847328186035, 4.817783355712891, 4.9171, 5.195, 5.3607, 4.6827, 5.035, 4.9048, 4.5954, 4.46, 4.625881195068359, 4.820620536804199, 4.808809280395508, 4.532168865203857, 4.3859944343566895, 4.64047384262085, 4.688878536224365, 4.438120365142822]


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(mae, label='training loss')
plt.plot(val_mae, label='validation loss')
plt.legend()
plt.show()


# In[ ]:




training.freeze(15)
history=training.fit_onecycle(1e-4,5,checkpoint_folder='/content/drive/My Drive/DL_Project/weights/new/')


# In[ ]:


#freezing first 15 layers and training the remaining layers
training.freeze(15)

training.fit_onecycle(1e-4,18,checkpoint_folder='/content/drive/MyDrive/DL_project/')


# In[ ]:


mae=[9.7028, 7.9876, 6.9388, 6.1731, 5.9146, 5.43145751953125, 5.4384589195251465, 5.487170219421387, 5.204601764678955, 4.83113431930542,4.7636 ,4.9500,5.0877,
     4.7607 ,4.8367 ,4.9354 ,4.7024 ,4.3650 ]
val_mae=[8.0117, 7.5809, 5.9674, 6.3078, 5.6124, 5.449876308441162, 5.407055854797363, 5.49222469329834, 5.0487847328186035, 4.817783355712891,4.9171,5.1950,5.3607,
         4.6827,5.0350,4.9048,4.5954,4.4600]


# In[ ]:


mae.extend(history.history['mae'])
val_mae.extend(history.history['val_mae'])


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(mae, label='training loss')
plt.plot(val_mae, label='validation loss')
plt.legend()
plt.show()


# In[ ]:


print(len(mae))


# In[ ]:


print(len(val_mae))


# In[ ]:


history1=training.fit_onecycle(1e-4,3,checkpoint_folder='/content/drive/My Drive/DL_Project/weights/new/2/')


# In[ ]:


training.get_weight_decay()


# In[ ]:


mae.extend(history1.history['mae'])
val_mae.extend(history1.history['val_mae'])


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(mae, label='training loss')
plt.plot(val_mae, label='validation loss')
plt.legend()
plt.show()


# In[ ]:


print(mae)
print(val_mae)


# In[ ]:


#saving the model
prediction.save('/content/drive/My Drive/DL_project/prediction')


# In[ ]:





# In[ ]:


history2=training.fit_onecycle(1e-4,4,checkpoint_folder='/content/drive/My Drive/DL_Project/weights/new/3/')


# In[ ]:


mae=[9.7028, 7.9876, 6.9388, 6.1731, 5.9146, 5.43145751953125, 5.4384589195251465, 5.487170219421387, 5.204601764678955, 4.83113431930542, 4.7636, 4.95, 5.0877, 4.7607, 4.8367, 4.9354, 4.7024, 4.365, 4.3475799560546875, 4.521537780761719, 4.697944164276123, 4.4693803787231445, 4.117332935333252, 4.131679534912109, 4.466404438018799, 4.104224681854248,3.9710 ,4.2446 ,4.2780 ,3.8742 ]
val_mae=[8.0117, 7.5809, 5.9674, 6.3078, 5.6124, 5.449876308441162, 5.407055854797363, 5.49222469329834, 5.0487847328186035, 4.817783355712891, 4.9171, 5.195, 5.3607, 4.6827, 5.035, 4.9048, 4.5954, 4.46, 4.625881195068359, 4.820620536804199, 4.808809280395508, 4.532168865203857, 4.3859944343566895, 4.64047384262085, 4.688878536224365, 4.438120365142822,4.5950,4.7407,4.6225,4.4619]


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.figure(figsize=(10,5))
plt.plot(mae, label='training loss')
plt.plot(val_mae, label='validation loss')
plt.legend()
plt.show()


# In[ ]:


print(mae)
print(val_mae)


# ## Prediction

# In[ ]:


prediction_1=ktrain.get_predictor(training.model,preprocess)


# In[ ]:


#function to predict the age
def predict_Age1(lis):
  pred=[]
  actual=[]
  for n in lis:
    fname=img_dir  +n
    #vis.show_image(fname)
    pred.append(round(prediction_1.predict_filename(fname)[0]))
    actual.append(int (p.search(fname).group(1)))
    #print('Predicted age : %s , Actual age : %s'%(pred,actual))
    
  return pred,actual


# In[ ]:


preds,actual=predict_Age1(test_data.filenames[0:2500])


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


cum_scores = cum_score(20, predictions=preds, y_test=actual)


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(np.arange(21), cum_scores)
plt.xticks(np.arange(21))
plt.show()


# In[ ]:


df_cs = pd.Series(cum_scores)


# In[ ]:


df_cs.to_csv('cum_score_CACD_2500.csv')


# In[ ]:


#function to predict the age
def predict_Age1(lis):
  pred=[]
  actual=[]
  for n in lis:
    fname=img_dir  +n
    #vis.show_image(fname)
    pred.append(round(prediction.predict_filename(fname)[0]))
    actual.append(int (p.search(fname).group(1)))
    #if(np.abs(round(prediction.predict_filename(fname)[0]) - int (p.search(fname).group(1))) == 0):
     # print(fname)
      #img=image.load_img(fname,target_size=(224,224))
      
      #img_ten=image.img_to_array(img)
      #img_ten=np.expand_dims(img_ten,axis=0)
      #img_ten/=255.""
      #img_ten.shapeplt.imshow((fname)[0])
      #plt.imshow(img)
      #print('Predicted age : %s , Actual age : %s'%(pred,actual))
    
  return pred,actual


# In[ ]:


preds,actual=predict_Age1(test_data.filenames[0:2500])


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

cum_scores = cum_score(20, predictions=preds, y_test=actual)
plt.figure(figsize=(9,6))
plt.plot(np.arange(21), cum_scores)
plt.xticks(np.arange(21))
plt.show()


# In[ ]:


df_cs = pd.Series(cum_scores)

df_cs.to_csv('cum_score_CACD_2500_no_Aug.csv')


# In[ ]:


no_aug=pd.read_csv('/content/drive/MyDrive/DL_Project/update/cum_score_CACD_2500_no_Aug.csv')
aug=pd.read_csv('/content/drive/MyDrive/DL_Project/update/cum_score_CACD_2500_Augumentation.csv')


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(np.arange(21), list(aug.iloc[:, 1]),label='With augmentation')
plt.plot(np.arange(21), list(no_aug.iloc[:, 1]),label='Without augmentation')
plt.xticks(np.arange(21))
plt.xlabel('Error')
plt.ylabel('Percentage')
plt.title('Cumulative error distribution of predictions on 2500 images')
plt.legend()
plt.show()


# In[ ]:


#saving the model
prediction.save('/content/drive/MyDrive/DL_project/prediction')


# ## Filter Check

# In[ ]:


#Loading a image
img_loc='/content/drive/MyDrive/DL_project/CACD_centered/photo.jpg'
img=image.load_img(img_loc,target_size=(224,224))
img_ten=image.img_to_array(img)
img_ten=np.expand_dims(img_ten,axis=0)
img_ten/=255.
img_ten.shape


# In[ ]:


layer_outputs=[layer.output for layer in model.layers[:15]]


# In[ ]:


from keras import models
activation_model=models.Model(inputs=model.input,outputs=layer_outputs)


# In[ ]:


activations=activation_model.predict(img_ten)


# In[ ]:


len(activations)


# In[ ]:


activation_model.summary()


# In[ ]:


first_cnn_op=activations[2]


# In[ ]:


print(first_cnn_op.shape)


# In[ ]:


plt.matshow(first_cnn_op[0,:,:,39],cmap='viridis')


# In[ ]:


layer = model.layers 

filters, biases = model.layers[2].get_weights()
print(layer[2].name, filters.shape)

   
# plot filters

fig1=plt.figure(figsize=(6, 6))
fig1.tight_layout()
columns = 8
rows = 8
n_filters = columns * rows
for i in range(1, n_filters +1):
    f = filters[:, :, :, i-1]
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([])
    plt.imshow(f[:, :, 1], cmap='viridis') #Show only the filters from 0th channel (R)
    #ix += 1
plt.show()    


# In[ ]:




#### Now plot filter outputs    

#Define a new truncated model to only include the conv layers of interest

conv_layer_index = [4, 18,49,70,132]  #TO define a shorter model
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = models.Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

#Input shape to the model is 224 x 224. SO resize input image to this shape.
from keras.preprocessing.image import load_img, img_to_array
img = load_img(img_loc, target_size=(224, 224)) #VGG user 224 as input

# convert the image to an array
img = img_to_array(img)
# expand dimensions to match the shape of model input
img = np.expand_dims(img, axis=0)

# Generate feature output by predicting on the input image
feature_output = model_short.predict(img)



for ftr in feature_output:
    #pos = 1
    fig=plt.figure(figsize=(2, 2))
    fig =plt.subplot(1,1,1)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(ftr[0, :, :, 13], cmap='viridis')
    plt.show()


# ## Comparing Augumented and without Augument models

# In[ ]:


train_mae_1=np.array([9.4393,7.3831,6.4295,5.2307,4.5213,4.0936 ,3.9339,3.8288,3.7833,3.7548,3.7087,3.3430 ,3.1496, 
                    3.0394 ,2.9743, 2.9433 ,2.8982 ,2.8402,2.7971,2.7520])


# In[ ]:


valid_mae_1=np.array([7.5368,7.1454,5.8458, 5.6703,5.5842,5.5307,5.5593,5.6940,5.7193,5.6905,5.5768,5.2224,
                    5.1577,5.1508,5.1649,5.1419,5.1424,5.1448,5.1449,5.1415])


# In[ ]:


mae=[9.7028, 7.9876, 6.9388, 6.1731, 5.9146, 5.43145751953125, 5.4384589195251465, 5.487170219421387, 5.204601764678955, 4.83113431930542, 4.7636, 4.95, 5.0877, 4.7607, 4.8367, 4.9354, 4.7024, 4.365, 4.3475799560546875, 4.521537780761719, 4.697944164276123, 4.4693803787231445, 4.117332935333252, 4.131679534912109, 4.466404438018799, 4.104224681854248,3.9710 ,4.2446 ,4.2780 ,3.8742 ]
val_mae=[8.0117, 7.5809, 5.9674, 6.3078, 5.6124, 5.449876308441162, 5.407055854797363, 5.49222469329834, 5.0487847328186035, 4.817783355712891, 4.9171, 5.195, 5.3607, 4.6827, 5.035, 4.9048, 4.5954, 4.46, 4.625881195068359, 4.820620536804199, 4.808809280395508, 4.532168865203857, 4.3859944343566895, 4.64047384262085, 4.688878536224365, 4.438120365142822,4.5950,4.7407,4.6225,4.4619]


# In[ ]:


x_1=np.linspace(0,29,30)


# In[ ]:


fig, ax=plt.subplots(2,1,figsize=(8,8))
ax[0].plot(x,train_mae_1,marker='*',label='training loss');
ax[0].plot(x,valid_mae_1,marker='*',label='validation loss');
ax[1].plot(x_1,mae,marker='*',label='training loss');
ax[1].plot(x_1,val_mae,marker='*',label='validation loss');
ax[0].set_xlim(0,20)
ax[0].set_ylim(2,10)
ax[1].set_xlim(0,30)
ax[1].set_ylim(2,10)

#ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("MAE")
ax[0].set_title("Model loss without Augumentation")
ax[0].legend(loc='best')

ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("MAE")
ax[1].set_title("Model loss with Augumentation")
ax[1].legend(loc='best')
fig.tight_layout()
#ax[0].show()


# ## Real Time Prediction

# In[ ]:


from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


# In[ ]:


from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))


# In[ ]:


# replace this image directory with yours
from PIL import Image
path = '/content/drive/MyDrive/DL_project/pics/' 

# directory path
orig_path = path

# folder to store the image
out_path = os.path.join(path, '/content/drive/MyDrive/DL_project/CACD_centered')

keep_picture=[]
if not os.path.exists(orig_path):
    raise ValueError(f'Original image path {orig_path} does not exist.')

if not os.path.exists(out_path):
    os.mkdir(out_path)

# looping through the images in your image directory

for picture_name in os.listdir(orig_path):
    stream = open(os.path.join(orig_path, picture_name), "rb") #read binary
    bytes = bytearray(stream.read()) #store the bytearray
    numpyarray = np.asarray(bytes, dtype=np.uint8) #convert to numpy array
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)  #decode and open the image using cv2
    stream.close()
    
        
    #img = cv2.imread(os.path.join(orig_path, picture_name))
    
    
    #fiding face locations
    print(picture_name)
    faces=face_recognition.face_locations(img)
        
    #iterating each face
    for face in faces:
        #file name with face location
        
        if len(faces) !=1:
            continue
        
        t = face[0]  # top location
        r = face[1]  # right location
        b = face[2]  # bottom location
        l = face[3]  # left location


        # just checking Raschka code from here
        width = r - l # width is right minus left
        height = b - t # height is bottom minus top
        diff =  height-width
        tol = 15 # additional increment
        up_down = 5

                
        if(diff > 0):#if height is greater than width -> adding access dimension left and right and moving up by 5 + tol
            if not diff % 2:  # if symmetric
                t=t-tol-up_down 
                b=b+tol-up_down
                l=l-tol-int(diff/2)
                r=r+tol+int(diff/2)
                tmp = img[(t):(b),(l):(r),:]
            else:
                t=t-tol-up_down
                b=b+tol-up_down
                l=l-tol-int((diff-1)/2)
                r=r+tol+int((diff+1)/2)
                tmp = img[(t):(b),(l):(r),:]
        if(diff <= 0):#if width is greater than hight -> adding access dimension to up and down and moving up by 5 +tol
            if not diff % 2:  # symmetric
                t=t-tol-int(diff/2)-up_down
                b=b+tol+int(diff/2)-up_down
                l=l-tol
                r=r+tol
                tmp = img[(t):(b),(l):(r),:]
            else:
                t=t-tol-int((diff-1)/2)-up_down
                b=b+tol+int((diff+1)/2)-up_down
                l=l-tol
                r=r+tol
                tmp = img[(t):(b),(l):(r),:]
    
        
        try:
            tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((224, 224), Image.ANTIALIAS))
            #A UINT8 is an 8-bit unsigned integer (range: 0 through 255 decimal). 
            #Because a UINT8 is unsigned, its first bit (Most Significant Bit (MSB)) is not reserved for signing
            
            #anti-aliasing is a technique for minimizing the distortion artifacts known as aliasing 
            #when representing a high-resolution image at a lower resolution
            
            #This function converts a numerical (integer or float) numpy array of any size and dimensionality into a image
            cv2.imwrite(os.path.join(out_path, picture_name), tmp)
            print(f'Wrote {picture_name}')
            keep_picture.append(picture_name)
            
        except ValueError:
            print(f'Failed {picture_name}')
            pass       


# In[ ]:


def real_time_pred(fname):
    pred=round(prediction_1.predict_filename(fname)[0])
    vis.show_image(fname)
    print('Predicted age : %s'%(pred))  


# In[ ]:





# In[ ]:


real_time_pred('/content/drive/MyDrive/DL_project/CACD_centered/photo.jpg')


# ## Finding loss

# In[ ]:


model = load_model("/content/drive/My Drive/DL_Project/CACD_preprocess/Prediction/tf_model.h5")
preproc = pickle.load(open("/content/drive/My Drive/DL_Project/CACD_preprocess/Prediction/tf_model.preproc",'rb'))

prediction = ktrain.get_predictor(model,preproc)


# In[ ]:


#predictor variable
prediction=ktrain.get_predictor(training.model,preprocess)


# In[ ]:


#function to predict the age
def predict_Age(lis):
  for n in lis:
    fname=img_dir  +n
    vis.show_image(fname)
    pred=round(prediction.predict_filename(fname)[0])
    actual = int (p.search(fname).group(1))
    print('Predicted age : %s , Actual age : %s'%(pred,actual))


# In[ ]:


predict_Age(test_data.filenames[0:5])


# In[ ]:


#function to predict the age
def predict_Age(lis):
    fname=lis
    vis.show_image(fname)
    pred=round(prediction.predict_filename(fname)[0])
    actual = int (p.search(fname).group(1))
    print('Predicted age : %s , Actual age : %s'%(pred,actual))

  


# In[ ]:


training.view_top_losses(n=5)


# In[ ]:


training.view_top_losses(n=3)


# In[ ]:


predict_Age('/content/CACD_centered/60_Shohreh_Aghdashloo_0005.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/54_Kelly_McGillis_0009.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/47_Traylor_Howard_0014.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/60_Tim_Gunn_0003.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/30_Hilarie_Burton_0008.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/40_Julian_Assange_0005.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/24_Summer_Glau_0016.jpg')


# In[ ]:


predict_Age('/content/CACD_centered/51_Carey_Lowell_0009.jpg')


# In[ ]:


#function to predict the age
def predict_Age1(lis):
  pred=[]
  actual=[]
  for n in lis:
    fname=img_dir  +n
    #vis.show_image(fname)
    pred.append(round(prediction.predict_filename(fname)[0]))
    actual.append(int (p.search(fname).group(1)))
    if(np.abs(round(prediction.predict_filename(fname)[0]) - int (p.search(fname).group(1))) >= 20):
      print(fname)
      img=image.load_img(fname,target_size=(224,224))
      
      #img_ten=image.img_to_array(img)
      #img_ten=np.expand_dims(img_ten,axis=0)
      #img_ten/=255.""
      #img_ten.shapeplt.imshow((fname)[0])
      plt.imshow(img)
      #print('Predicted age : %s , Actual age : %s'%(pred,actual))
    
  return pred,actual

preds,actual=predict_Age1(test_data.filenames[0:250])


# ## References

# https://arxiv.org/pdf/1512.03385v1.pdf

# https://arxiv.org/ftp/arxiv/papers/1709/1709.01664.pdf
