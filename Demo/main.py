

# ## **Loading required packages, models, functions**

get_ipython().system('pip install ktrain')
get_ipython().system('pip install -U kora')
get_ipython().system('pip install ffmpeg-python')
get_ipython().system('pip install tensorflow_addons')


import numpy as np
import cv2
import os, ffmpeg
from tensorflow import keras
from tensorflow.keras.models import load_model
import ktrain
import time
import pickle
from datetime import datetime
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import HTML
from base64 import b64encode
import imutils
from kora.drive import upload_public
import sys
from subprocess import run, PIPE
from pathlib import Path
import tensorflow_addons as tfa

#Importing the Haar Cascades classifier XML file.
face_cascade = cv2.CascadeClassifier("DL_Age_Detection/Final_Model/haarcascade_frontalface_default.xml")



# #Loading the trained model and the preprocessed model

model1 = load_model("DL_Age_Detection/Final_Model/tf_model_with_weight_decay.h5",compile=False)
preproc = pickle.load(open("DL_Age_Detection/Final_Model/tf_model_with_weight_decay.preproc",'rb'))


opt=tfa.optimizers.AdamW(weight_decay=10**-3,learning_rate=10**-4 )
model1.compile(loss='MAE',optimizer=opt)
prediction = ktrain.get_predictor(model1,preproc)


#Defining a function to shrink the detected face region by a scale for better prediction in the model.
def shrink_face_roi(x, y, w, h, scale=0.8):
    wh_multiplier = (1-scale)/2
    x_new = int(x + (w * wh_multiplier))
    y_new = int(y + (h * wh_multiplier))
    w_new = int(w * scale)
    h_new = int(h * scale)
    return (x_new, y_new, w_new, h_new)


#Defining a function to create the predicted age overlay on the image by centering the text.
def create_age_text(img, age_val, x, y, w, h):

    # Defining font, scales and thickness.
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 2

    text = str(age_val)

    text = text + " " + "years"

    # Getting width, height and baseline of age text and "years old".
    (text_width, text_height), text_bsln = cv2.getTextSize(text, fontFace=fontFace, fontScale=text_scale, thickness=2)

    # Calculating center point coordinates of text background rectangle.
    x_center = x + (w/2)
    y_text_center = y + h + 40

    # Calculating bottom left corner coordinates of text based on text size and center point of background rectangle calculated above.
    x_text_org = int(round(x_center - (text_width / 2)))
    y_text_org = int(round(y_text_center + (text_height / 2)))

    face_age_background = cv2.rectangle(img, (x-1, y+h), (x+w+1, y+h+94), (0, 120, 0), cv2.FILLED)
    face_age_text = cv2.putText(img, text, org=(x_text_org, y_text_org), fontFace=fontFace, fontScale=text_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)

    return (face_age_background, face_age_text)


#Defining a function to find faces in an image and then determine the age for each face found in the image.
def determine_age(img):

    # Making a copy of the image for overlay of ages and making a grayscale copy for passing to the loaded model.
    img_copy = np.copy(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the image using the face_cascade loaded above and storing their coordinates into a list.
    faces = face_cascade.detectMultiScale(img_copy, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)) 

      # Looping through each face found in the image.
    for i, (x, y, w, h) in enumerate(faces):

        # Drawing a rectangle around the found face.
        face_rect = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 100, 0), thickness=2)
        
        # Predicting the age of the found face using the model loaded above.
        x2, y2, w2, h2 = shrink_face_roi(x, y, w, h)
        #x2, y2, w2, h2 = x,y,w,h
        face_roi = img_gray[y2:y2+h2, x2:x2+w2]
        face_roi = cv2.resize(face_roi, (224, 224))
        cv2.imwrite(os.path.join("DL_Age_Detection/Demo", "runtime_picture.jpg"), face_roi)
        face_age = round(prediction.predict_filename("DL_Age_Detection/Demo/runtime_picture.jpg")[0])
        os.remove("runtime_picture.jpg")
        
        # Calling the above defined function to create the predicted age overlay on the image.
        face_age_background, face_age_text = create_age_text(img_copy, face_age, x, y, w, h)

    return img_copy


#Function to save the image with age in the same location
def new_img_name(org_img_path):
    img_path, img_name_ext = os.path.split(org_img_path)
    img_name, img_ext = os.path.splitext(img_name_ext)

    new_img_name_ext = img_name+"_WITH_AGE"+img_ext
    new_img_path = os.path.join(img_path, new_img_name_ext)

    return new_img_path

#Function to determine age from image
def determine_age_from_image(my_image):
  img = cv2.imread(my_image)
  age_img = determine_age(img)

  # Saving the new generated image with a new name at the same location. 
  try:
    new_my_image = new_img_name(my_image)
    cv2.imwrite(new_my_image, age_img)
    print(f"Saved to {new_my_image}")
  except:
    print("Error: Could not save image!")

  return(age_img)


#Function to open webcam, capture image and determine age
def determine_age_from_webcam(filename='my_picture_webcam.jpg', quality=0.8):
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
  
  webcam_image_with_age = determine_age_from_image(filename)
  os.remove(filename)

  return webcam_image_with_age


#Function to save video with age at the same location
def new_vid_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_vid_name_ext = vid_name+"_WITH"+".mp4"
    new_vid_path = os.path.join(vid_path, new_vid_name_ext)

    return new_vid_path


#Function to determine age from a video
def determine_age_from_video(my_video):
  # Creating a VideoCapture object.
  cap = cv2.VideoCapture(my_video)

  # Checking if video can be accessed successfully.
  if (cap.isOpened() == False): 
    print("Unable to read video!")

  # Getting the video frame width and height.
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))

  # Defining the codec and creating a VideoWriter object to save the output video at the same location.
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  new_my_video = new_vid_name(my_video)
  out = cv2.VideoWriter(new_my_video, fourcc, 18, (frame_width, frame_height))

  while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        # Running age detection on the grabbed frame.
        age_img = determine_age(frame)
        
        # Saving frame to output video using the VideoWriter object defined above.
        out.write(age_img)

    else:
        break

  #Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
  cap.release()
  out.release()
  cv2.destroyAllWindows()

  #Compressing the video
  args = sys.argv[1:]
  video_file = Path(' '.join(args), new_my_video)
  run(['ffmpeg', '-i', video_file.name, '-vcodec', 'h264', '-acodec','aac', video_file.name.replace('.' + video_file.name.split('.')[-1], '_AGE.' + video_file.name.split('.')[-1])])
  os.remove(new_my_video)

  print(f"Saved to my_video_WITH_AGE.mp4")


# ## **Age Determination on Image**
# Provide the image filepath as a string below.
my_image = "my_picture.jpg"

image_with_age = determine_age_from_image(my_image)

try:
  cv2_imshow(image_with_age)
except:
  print("")

# ## **Age Determination on Video**
# Provide the video filepath as a string below
my_video = "my_video.mp4"

determine_age_from_video(my_video)

#In order to play the video
url = upload_public('my_video_WITH_AGE.mp4')
HTML(f"""<video src={url} width=500 controls/>""")


# ## **Age Determination on Webcam**
image_with_age_webcam = determine_age_from_webcam()

cv2_imshow(image_with_age_webcam)
cv2.waitKey(0);


# ## **References**

# 1. https://colab.research.google.com/github/dortmans/ml_notebooks/blob/master/face_detection.ipynb#scrollTo=5WICWY6_7p6b
# 2. https://towardsdatascience.com/age-detection-using-facial-images-traditional-machine-learning-vs-deep-learning-2437b2feeab2
# 3. https://dev.to/m4cs/compressing-videos-easily-on-windows-w-ffmpeg-and-registry-files-5fin
