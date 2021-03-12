import streamlit as st
# import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)
from skimage import *
# from skimage.io import imread
from skimage.transform import resize
import matplotlib as mpl
import pandas as pd
import skimage

# import cv2
import pandas as pd
import os
# import wget
from imageai.Detection import ObjectDetection

from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale

import pickle

from PIL import Image
from numpy import asarray
from skimage import transform
from skimage import exposure
from skimage import io
import requests
import tensorflow as tf
import keras

st.title("Traffic Sign Recognition using Image Classification")
st.title("Team Name: AI Dragon Drivers")
drexel = Image.open(requests.get("https://i.pinimg.com/originals/56/64/72/56647273a66b7e3d919ae9908160300e.jpg", stream=True).raw)
st.image(drexel, width=200)

input_task = st.selectbox("Select task", options=list(['Detection_Classification','Classification']))
# st.write(input_task)
st.write("[Example link](https://s.driving-tests.org/wp-content/uploads/2019/10/stop-sign-2.jpg)")
url = st.text_input("Input the image URL here")
# image_path = Image.open(requests.get(url, stream=True).raw)
# image_path.save("imagefromurl.jpg")
# image_path = "imagefromurl.jpg"

def urltoimage(url):
    
    image_path = Image.open(requests.get(url, stream=True).raw)
    image_path.save("imagefromurl.jpg")
    image_path = "imagefromurl.jpg"
    return image_path

# Loading Object Detection
object_detector = ObjectDetection()
model_path = os.getcwd()
object_detector = ObjectDetection()
object_detector.setModelTypeAsRetinaNet()
# object_detector.setModelPath( os.path.join(model_path , "resnet50_coco_best_v2.0.1.h5"))

# path = 'C:\\Users\\laksh\\Documents\\Lakshmikanth\\Drexel Classes\\Term 6\\Capstone Project 2\\Object Detection\\Full_data'
object_detector.setModelTypeAsTinyYOLOv3() #detector.setModelTypeAsRetinaNet() detector.setModelPath( os.path.join(execution_path , "model/yolo-tiny.h5"))
# object_detector.setModelPath( os.path.join(path , "yolo-tiny.h5"))
object_detector.setModelPath( "yolo-tiny.h5")
object_detector.loadModel()

def detect_object(image_path):
    object_detections = object_detector.detectObjectsFromImage(input_image=os.path.join(model_path , "imagefromurl.jpg"), output_image_path=os.path.join(model_path , "image1_new.jpg"))
    stop_points = ''
    if object_detections!=[]:
        
        for detection in object_detections:
            if detection["name"]=='stop sign':
                stop_points = detection["box_points"]
                print(detection["box_points"])
    else:
        image = io.imread(image_path)
        
        stop_points = [0,0,image.shape[0],image.shape[1]]
        

    st.write("Actual Image is :")
    image = io.imread(image_path)
    st.image(image, width=None)
    
    return stop_points

newimage_path = "image1_new.jpg"

def crop_image(newimage_path, box_points):
    image = io.imread(newimage_path)
    image = Image.fromarray(image) 
    image = image.crop((box_points[0],box_points[1],box_points[2],box_points[3]))
    image = image.save("image_cropped.jpg")
    st.write("Detected Image is :")

    image = io.imread(newimage_path)
    st.image(image, width=None)
    print('New Image is saved as "image_cropped.jpg"')

cropped_image = "image_cropped.jpg"    

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X,orientations=self.orientations,pixels_per_cell=self.pixels_per_cell,cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])
        
        
# Load the model and pipeline from disk        
import joblib
sgd_clf = joblib.load('sgd_clf_full_data.pkl')
cnn_model = keras.models.load_model("CNN_full_data.h5")
# Function for Classification of image


def classify_image(image_path):
    
    data = Image.open(image_path)
    data = asarray(data)
    image = transform.resize(data, (80, 80))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)


    data = Image.open(image_path)
    data = asarray(data)
    image1 = transform.resize(data, (80, 80))
    image1 = exposure.equalize_adapthist(image1, clip_limit=0.1)

    data = []
    data.append(image)
    data.append(image1)
    data = np.array(data)
    
#     image1 = pipeline_fitted.transform(data)
#     image1 = grayify.transform(data)
#     image1 = hogify.transform(image1)
#     image1 = scalify.transform(image1)

    predictions = sgd_clf.predict(data[1:2])
    st.write("Prediction of SGD model is: ",predictions[0])

    cnn_pred = cnn_model.predict(data[1:2])
#     st.write("Prediction of SVC model is: ",predictions[0])
    
    score = tf.nn.softmax(cnn_pred[0])
  
    
#     st.write("Model predicts the image as: ")
    
#     md_results = f"## CNN Model predicts the image as: **{label_df['labels_cat'][label_df.labels==np.argmax(score)].head(1).iloc[0]}** ##\n With Confidence of **{100 * np.max(score)}**."

#     st.markdown(md_results)
    st.write("Prediction of CNN model is: ",label_df['labels_cat'][label_df.labels==np.argmax(score)].head(1).iloc[0])
    image = io.imread(image_path)
    st.image(image, width=None)
    
    
def full_detect(url):   
    image_path = urltoimage(url)
#     st.write('1')
    box_points = detect_object(image_path)
#     st.write('2')
    crop_image(newimage_path, box_points)    
#     st.write('3')
    classify_image(cropped_image)
#     st.write('4')
    
def classification(url):
    image_path = urltoimage(url)
    data = Image.open(image_path)
    data = asarray(data)
    image = transform.resize(data, (80, 80))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)


    data = Image.open(image_path)
    data = asarray(data)
    image1 = transform.resize(data, (80, 80))
    image1 = exposure.equalize_adapthist(image1, clip_limit=0.1)

    data = []
    data.append(image)
    data.append(image1)
    data = np.array(data)


    predictions = sgd_clf.predict(data[1:2])
    cnn_pred = cnn_model.predict(data[1:2])
    st.write("Prediction of SVC model is: ",predictions[0])
#     st.write("Prediction of CNN model is: ",cnn_pred[0])
    
    score = tf.nn.softmax(cnn_pred[0])
  
    
#     st.write("Model predicts the image as: ")
    
#     md_results = f"## CNN Model predicts the image as: **{label_df['labels_cat'][label_df.labels==np.argmax(score)].head(1).iloc[0]}** ##\n With Confidence of **{100 * np.max(score)}**."

#     st.markdown(md_results)
    st.write("Prediction of CNN model is: ",label_df['labels_cat'][label_df.labels==np.argmax(score)].head(1).iloc[0])
    
    image = io.imread(image_path)
    st.image(image, width=None)
    
    print('Prediction of SGD modeel is: ',predictions)
    plt.imshow(data[1], cmap=mpl.cm.binary)
    plt.axis("off")
    plt.show()
    
label_df = pd.read_csv("label_df.csv") 
if input_task == 'Detection_Classification':     
    
    try:    
        full_detect(url)
    except ValueError:
        st.error("Please enter a valid input which is in JPG format")
else:
    
    try:    
        classification((url))
    except ValueError:
        st.error("Please enter a valid input which is in JPG format")   
