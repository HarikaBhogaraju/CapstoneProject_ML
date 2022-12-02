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

#load the datasets
x_train = setup.X_TRAIN
y_train = setup.Y_TRAIN
x_test = setup.X_TEST
y_test = setup.Y_TEST

#filename of saved model
filename = '/Users/harikabhogaraju/CSE486_ML/capstone/XGBoost_EcoFilter_pck.h5'

#load the saved model
xgb_model_latest = xgb.XGBClassifier()
xgb_model_latest._le = LabelEncoder().fit(y_test)
xgb_model_latest = pickle.load(open(filename, 'rb'))

"""## Classification Report on Test Set"""
from sklearn.metrics import classification_report
import xgboost


SIZE = x_test[0].shape[0]

#Load model without classifier/fully connected layers
#resnetModel = ResNet50(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')
VGG_model = VGG16(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')

for layer in VGG_model.layers:
  layer.trainable = False
#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
#resnetModel.trainable = False

#VGG_model.summary()  #Trainable parameters will be 0
xgb_model_latest = pickle.load(open(filename, 'rb'))#Now, let us use features from convolutional network for RF
x_teat_features = VGG_model.predict(x_test)
x_test_features = x_teat_features.reshape(x_teat_features.shape[0], -1)

y_pred = xgb_model_latest.predict(x_test_features)

target_names = ['class 0(Dirty)', 'class 1(Clean)']



#THIS TEST IS DONE FROM AFTER ECOFILTER IMAGES
print(classification_report(y_test, y_pred, target_names=target_names))

x_test_features.shape

"""## Testing on EColi - before filtering only"""

dirtyImagesTest = setup.TestDirty #can also add custom dataset
cleanImagesTest = setup.TestClean #can also add custom dataset

dirtyImagesTest_c = setup.extract_image_center(dirtyImagesTest, hcrop=20, wcrop=20)
cleanImagesTest_c = setup.extract_image_center(cleanImagesTest, hcrop=20, wcrop=20)

plt.subplot(3,2,1)
plt.imshow(dirtyImagesTest_c[0])
plt.subplot(3,2,2)
plt.imshow(setup.TestDirty[0])

plt.subplot(3,2,3)
plt.imshow(cleanImagesTest_c[0])
plt.subplot(3,2,4)
plt.imshow(setup.TestDirty[-1])

images_np_array_dirty = np.array(dirtyImagesTest_c)
y_true_dirty = np.array([0 for i in range(images_np_array_dirty.shape[0])])
images_np_array_clean = np.array(cleanImagesTest_c)
y_true_clean = np.array([1 for i in range(images_np_array_clean.shape[0])])

images_np_array = np.concatenate((images_np_array_dirty, images_np_array_clean), axis=0)

y_true = np.concatenate((y_true_dirty, y_true_clean), axis=0)

SIZE = images_np_array[0].shape[0]

#Load model without classifier/fully connected layers
#resnetModel = ResNet50(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')
VGG_model = VGG16(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
  layer.trainable = False
#resnetModel.trainable = False

#Now, let us use features from convolutional network for RF
feature_extractor = VGG_model.predict(x_train)
#Now, let us use features from convolutional network for RF
x_teat_features = VGG_model.predict(images_np_array)
x_test_features = x_teat_features.reshape(x_teat_features.shape[0], -1)

y_pred = xgb_model_latest.predict(x_test_features)
print(y_pred)
target_names = ['class 0(Dirty)', 'class 1(Clean)']
print(classification_report(y_true, y_pred, target_names=target_names))
