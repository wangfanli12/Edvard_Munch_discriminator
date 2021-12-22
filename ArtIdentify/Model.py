from keras.layers.normalization.batch_normalization import BatchNormalization

from PIL import Image
import csv
from numpy import asarray
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import tensorflow as tf

# This file contains the model and trains the model

def codeToNum(emotion):
    if emotion == "happiness":
        return 0
    elif emotion == "sadness":
        return 1
    elif emotion == "fear":
        return 2
    elif emotion == "anger":
        return 3
    return 4
    

def load_sample():
    result = open("pool_sum.tsv")
    
    read_tsv = csv.reader(result, delimiter="\t")
    count = 0
    
    data = 0
    label = 0
    root = "Edvard_Munch/"
    
    count = 0
    for row in read_tsv:
      if row[1] != "neutral":
          picture = row[0].replace('https://raw.githubusercontent.com/wangfanli12/Edvard_Munch/main/Edvard_Munch/', '')
          if picture != 'INPUT:image':
              image = Image.open(root + picture)
              image = image.resize((256, 256))
              if isinstance(data, int):
                  data = np.array([asarray(image)])
                  label = np.array([[codeToNum(row[2])]])
              else:
                  data = np.append(data,  np.array([asarray(image)]), axis=0)
                  label = np.append(label, np.array([[codeToNum(row[1])]]), axis=0)
              count = count + 1
              
    data = data.astype('float32')
    data = (data - 127.5) / 127.5
    result.close()
    label = to_categorical(label, 5)
    return data, label

def define_discriminator(in_shape=(256,256,3)):
    model = Sequential()

    model.add(Conv2D(256, (3,3), padding='same', input_shape=in_shape, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model

def train_discriminator(model, dataset, n_iter=20):
    for i in range(n_iter):
        X_real, y_real = load_sample()
        model.train_on_batch(X_real, y_real)
        
def train_model():
    data, label = load_sample()
    #print(data)
    #print(label)
    model = define_discriminator()
    train_discriminator(model, data, 30)
    model.save("saved_model")