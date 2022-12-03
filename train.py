import setup
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
from sklearn.model_selection import GridSearchCV

import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost
import xgboost as xgb
import random
from sklearn import metrics

X_TRAIN = []
X_TEST = []

Y_TRAIN = []
Y_TEST = []

TrainDirty = []
TestDirty = []

TrainClean = []
TestClean = []


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

def load_labels_from_file(file, resize=True):
    labels = []
    with open(file) as label_file:
        for line in label_file:
            print(line)
            labels.append(int(line))
    return labels

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

def setData(pathData):
    images = load_images_from_folder(pathData)
    return images

def setLabels(pathLabels):
    labels = load_labels_from_file(pathLabels)
    return labels

def load_data(pathData):
        dataset_clean = load_images_from_folder(pathData+'TrainClean')
        print("Data clean read")

        TrainClean = []
        for i in range(len(dataset_clean)):
            TrainClean.append(dataset_clean[i])
        print("Training Clean Data: ", len(TrainClean))

        dataset_dirty = load_images_from_folder(pathData+'TrainDirty')
        print("Data dirty read")

        TrainDirty = []
        for i in range(len(dataset_dirty)):
            TrainDirty.append(dataset_dirty[i])
        print("Training Dirty Data: ", len(TrainDirty))

        Clean_u_idx = np.random.choice(np.array([i for i in range(len(TrainClean))]), len(TrainDirty), replace=False)
        Clean_u = [TrainClean[i] for i in Clean_u_idx]

        dataset = TrainDirty + Clean_u

        dataset_c = extract_image_center(dataset, hcrop=40/2, wcrop=40/2)
        labels = [0 for i in range(len(TrainDirty))]+[1 for i in range(len(Clean_u))]

        #randomize the list to ensure batches get mix of clean and dirty images
        mapIndexPosition = list(zip(dataset_c, labels))
        random.shuffle(mapIndexPosition)
        # make list separate
        images, labels = zip(*mapIndexPosition)
        return images, labels

def model_predict(x_train,y_train,x_test,y_test):
        num_classes = 2
        SIZE = x_train[0].shape[0]

        #Load model without classifier/fully connected layers
        VGG_model = VGG16(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')

        #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
        for layer in VGG_model.layers:
          layer.trainable = False
        #VGG_model.summary()  #Trainable parameters will be 0


        #Now, let us use features from convolutional network for RF
        feature_extractor = VGG_model.predict(x_train)

        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
        print(features.shape)

        #XGBOOST
        model = xgb.XGBClassifier()

        model.fit(features, y_train) #For sklearn no one hot encoding

        #Send test data through same feature extractor process
        X_test_feature = VGG_model.predict(x_test)
        X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

        #Now predict using the trained RF model.
        prediction = model.predict(X_test_features)
        return prediction, VGG_model, model

def train_main(pathData):
    #read datasets
    images, labels = load_data(pathData)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    x_train = np.array(x_train).astype('float32')
    x_test = np.array(x_test).astype('float32')

    print(Counter(y_train), Counter(y_test), x_train.shape)

    prediction, VGG_model, model = model_predict(x_train, y_train, x_test, y_test)
    #Print overall accuracy
    print ("Accuracy = ", metrics.accuracy_score(y_test, prediction))

    #Check results on a few select images
    n=np.random.randint(0, x_test.shape[0])
    img = x_test[n]
    plt.imshow(img)
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_feature = VGG_model.predict(input_img)
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction = model.predict(input_img_features)[0]

    print("The prediction for this image is: ", prediction)
    print("The actual label for this image is: ", y_test[n])

    # saving XGBoost model
    filename = 'models/XGBoost_EcoFilter_pck.h5'
    model_save(filename,model)

    # testing the saved and loaded model. loading the model now
    xgb_model_latest = xgb.XGBClassifier()
    xgb_model_latest = model_load(filename, y_test)

    """## Classification Report"""

    SIZE = x_test[0].shape[0]

    VGG_model = VGG16(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')

    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in VGG_model.layers:
      layer.trainable = False
    #VGG_model.summary()  #Trainable parameters will be 0

    xgb_model_latest = pickle.load(open(filename, 'rb')) #loading the model
    x_test_features_init = VGG_model.predict(x_test)
    x_test_features = x_test_features_init.reshape(x_test_features_init.shape[0], -1)

    y_pred = xgb_model_latest.predict(x_test_features)

    print_classification_report(y_pred,y_test)

def model_save(filename, model):
    pickle.dump(model, open(filename, 'wb'))
    model.save_model('models/XGBoost_EcoFilter.h5')

def model_load(filename, y_test):
    xgb_model_latest = xgb.XGBClassifier()
    xgb_model_latest._le = LabelEncoder().fit(y_test)
    xgb_model_latest = pickle.load(open(filename, 'rb'))
    return xgb_model_latest

def print_classification_report(y_pred,y_true):
    target_names = ['class 0(Dirty)', 'class 1(Clean)']
    print(classification_report(y_true, y_pred, target_names=target_names))
