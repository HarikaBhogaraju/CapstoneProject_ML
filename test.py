#Import all the libraries necessary
import setup
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout , Input , Flatten , Conv2D , MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping , ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.applications.vgg16 import VGG16
from PIL import Image

import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost
import xgboost as xgb
import random
from sklearn import metrics


X_TEST = []
Y_TEST = []

x_test = []
y_test = []

y_true = []

TestDirty = []
TestClean = []

images_np_array = []

def load_images_from_folder(folder, resize=True):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if resize:
            img = cv2.resize(img,(150,150))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images

def extract_image_center(images, hcrop=0, wcrop=10):
    image_centers = []
    for img in images:
        x = img.shape[0]
        y = img.shape[1]
        xc = int(hcrop*x/100)
        yc = int(wcrop*y/100)
        img = img[xc:x-xc, yc:y-yc]
        img = cv2.resize(img,(150,150))
        image_centers.append(img)
    return image_centers

def file_name(folder):
    file_name = []
    for filename in os.listdir(folder):
        file_name.append(filename)
    return file_name

def model_load(filename):

        #load the saved model
        xgb_model_latest = xgb.XGBClassifier()
        xgb_model_latest._le = LabelEncoder().fit(y_test)
        xgb_model_latest = pickle.load(open(filename, 'rb'))

        return xgb_model_latest

        #THIS TEST IS DONE FROM AFTER ECOFILTER IMAGES
        #print(classification_report(y_test, y_pred, target_names=target_names))
def model_predict(xgb_model_latest,data):
    SIZE = data[0].shape[0]

    #Load model without classifier/fully connected layers
    VGG_model = VGG16(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')

    for layer in VGG_model.layers:
        layer.trainable = False
    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights

    x_teat_features = VGG_model.predict(data)
    x_test_features = x_teat_features.reshape(x_teat_features.shape[0], -1)

    y_pred = xgb_model_latest.predict(x_test_features)

    target_names = ['class 0(Dirty)', 'class 1(Clean)']

    return y_pred

def load_data(pathData):
        dataset_clean = load_images_from_folder(pathData+'/TestClean')
        print("Data clean read")

        TestClean = []
        for i in range(len(dataset_clean)):
            TestClean.append(dataset_clean[i])
        print("Testing Clean Data: ", len(TestClean))

        dataset_dirty = load_images_from_folder(pathData+'/TestDirty')
        print("Data dirty read")

        TestDirty = []
        for i in range(len(dataset_dirty)):
            TestDirty.append(dataset_dirty[i])
        print("Testing Dirty Data: ", len(TestDirty))


        dataset = TestDirty + TestClean

        dataset_c = extract_image_center(dataset, hcrop=40/2, wcrop=40/2)
        labels = [0 for i in range(len(TestDirty))]+[1 for i in range(len(TestClean))]

        #randomize the list to ensure batches get mix of clean and dirty images
        mapIndexPosition = list(zip(dataset_c, labels))
        random.shuffle(mapIndexPosition)
        # make list separate
        images, labels = zip(*mapIndexPosition)
        return images, labels

def test_main(pathData):

    images, labels = load_data(pathData)

    filename = 'models/XGBoost_EcoFilter_pck.h5'
    xgb_model_latest = model_load(filename) #load model

    images_np_array = np.array(images)
    y_true = np.array(labels)

    y_pred = model_predict(xgb_model_latest,images_np_array) #get prediction

    print_classification_report(y_pred,y_true) #classification report

def print_classification_report(y_pred,y_true):
    target_names = ['class 0(Dirty)', 'class 1(Clean)']
    print(classification_report(y_true, y_pred, target_names=target_names))
