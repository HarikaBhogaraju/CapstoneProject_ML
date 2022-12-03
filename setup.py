
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
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
import random
from sklearn.model_selection import train_test_split
from PIL import Image

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

def setData():
    #for SA and AZ images
    CleanAZ = load_images_from_folder('/Users/harikabhogaraju/CSE486_ML/capstone/datasets/AZ_CleanBlobs') #Training
    #
    CleanSA = load_images_from_folder('/Users/harikabhogaraju/CSE486_ML/capstone/datasets/SA_CleanBlobs') #80% for training, 20% for testing
    #
    DirtyAZ = load_images_from_folder('/Users/harikabhogaraju/CSE486_ML/capstone/datasets/AZ_DirtyBlobs') #Training
    #
    DirtySA = load_images_from_folder('/Users/harikabhogaraju/CSE486_ML/capstone/datasets/SA_DirtyBlobs') #80% for training, 20% for testing
    #

    lenD = len(DirtySA)
    lenC = len(CleanSA)

    trainLenD = int(0.8*lenD)
    trainLenC = int(0.8*lenC)
    print("SA Train Dirty: ",trainLenD)
    print("SA Train Clean: ",trainLenC)
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
    print("AZ Train Dirty: ",lenD)
    print("AZ Train Clean: ",lenC)
    for i in range(lenD):
      TrainDirty.append(DirtyAZ[i])
    for i in range(lenC):
      TrainClean.append(CleanAZ[i])

    print(len(TrainDirty))
    print(len(TrainClean))
    print(len(TestClean))
    print(len(TestDirty))
    count = 0

#TRAINING DATA
    #class balance
    Clean_u_idx = np.random.choice(np.array([i for i in range(len(TrainClean))]), len(TrainDirty), replace=False)
    Clean_u = [TrainClean[i] for i in Clean_u_idx]

    dataset = TrainDirty + Clean_u

    dataset_c = extract_image_center(dataset, hcrop=40/2, wcrop=40/2)
    labels = [0 for i in range(len(TrainDirty))]+[1 for i in range(len(Clean_u))]

    # del Dirty
    # del Clean
    Counter(labels)

    #randomize the list to ensure batches get mix of clean and dirty images
    mapIndexPosition = list(zip(dataset_c, labels))
    random.shuffle(mapIndexPosition)
    # make list separate
    images, labels = zip(*mapIndexPosition)


    Counter(labels)

#TESTING
    #class balance
    #Clean_u_idx = np.random.choice(np.array([i for i in range(len(TestClean))]), len(TestDirty), replace=False)
    #Clean_u = [TestClean[i] for i in Clean_u_idx]
    count = 0
    dataset = TestDirty + TestClean

    dataset_c = extract_image_center(dataset, hcrop=40/2, wcrop=40/2)


    labels_test = [0 for i in range(len(TestDirty))]+[1 for i in range(len(Clean_u))]

    # del Dirty
    # del Clean
    Counter(labels)

    #randomize the list to ensure batches get mix of clean and dirty images
    mapIndexPosition = list(zip(dataset_c, labels_test))
    random.shuffle(mapIndexPosition)
    # make list separate
    images_test, labels_test = zip(*mapIndexPosition)
    Counter(labels_test)

setData()
