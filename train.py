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
import xgboost
import xgboost as xgb
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

def train_main():
    #for SA and AZ images
    CleanAZ = load_images_from_folder('datasets/AZ_CleanBlobs') #Training
    CleanSA = load_images_from_folder('datasets/SA_CleanBlobs') #80% for training, 20% for testing

    DirtyAZ = load_images_from_folder('datasets/AZ_DirtyBlobs') #Training
    DirtySA = load_images_from_folder('datasets/SA_DirtyBlobs') #80% for training, 20% for testing

    lenD = len(DirtySA)
    lenC = len(CleanSA)

    trainLenD = int(0.8*lenD)
    trainLenC = int(0.8*lenC)

    for i in range(trainLenD):
      TrainDirty.append(DirtySA[i]) #80% of DirtySA dataset
    for i in range(trainLenD,lenD):
      TestDirty.append(DirtySA[i]) #20% of DirtySA datset

    for i in range(trainLenC):
      TrainClean.append(CleanSA[i]) #80% of CleanSA dataset
    for i in range(trainLenC,lenC):
      TestClean.append(CleanSA[i]) #20% of CleanSA datset

    lenD = len(DirtyAZ)
    lenC = len(CleanAZ)
    for i in range(lenD):
      TrainDirty.append(DirtyAZ[i])
    for i in range(lenC):
      TrainClean.append(CleanAZ[i])

    print(len(TrainDirty))
    print(len(TrainClean))
    print(len(TestClean))
    print(len(TestDirty))


    #class balance
    Clean_u_idx = np.random.choice(np.array([i for i in range(len(TrainClean))]), len(TrainDirty), replace=False)
    Clean_u = [TrainClean[i] for i in Clean_u_idx]

    dataset = TrainDirty + Clean_u
    dataset_c = extract_image_center(dataset, hcrop=40/2, wcrop=40/2)
    labels = [0 for i in range(len(TrainDirty))]+[1 for i in range(len(Clean_u))]

    # del Dirty
    # del Clean
    Counter(labels)

    plt.subplot(3,2,1)
    plt.imshow(dataset_c[0])
    plt.subplot(3,2,2)
    plt.imshow(dataset[0])

    plt.subplot(3,2,3)
    plt.imshow(dataset_c[-1])
    plt.subplot(3,2,4)
    plt.imshow(dataset[-1])

    plt.subplot(3,2,5)
    plt.imshow(dataset_c[-10])
    plt.subplot(3,2,6)
    plt.imshow(dataset[-10])

    #randomize the list to ensure batches get mix of clean and dirty images
    import random
    mapIndexPosition = list(zip(dataset_c, labels))
    random.shuffle(mapIndexPosition)
    # make list separate
    images, labels = zip(*mapIndexPosition)
    Counter(labels)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    x_train = np.array(x_train).astype('float32')
    x_test = np.array(x_test).astype('float32')

    X_TRAIN = x_train
    X_TEST = x_test
    Y_TRAIN = y_train
    Y_TEST = y_test

    print(Counter(y_train), Counter(y_test), x_train.shape)

    #read datasets

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

    #Print overall accuracy
    from sklearn import metrics
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
    filename = 'XGBoost_EcoFilter_pck.h5'
    pickle.dump(model, open(filename, 'wb'))

    model.save_model('XGBoost_EcoFilter.h5')

    # testing the saved and loaded model. loading the model now

    xgb_model_latest = xgb.XGBClassifier()


    xgb_model_latest._le = LabelEncoder().fit(y_test)

    xgb_model_latest = pickle.load(open(filename, 'rb'))


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

    target_names = ['class 0(Dirty)', 'class 1(Clean)']

    print(classification_report(y_test, y_pred, target_names=target_names))
