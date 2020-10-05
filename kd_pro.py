from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import os, shutil

def build_model():
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(450, 450, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    return model

filepath = input("Set path ")
model = build_model()
model.load_weights("C:\\Users\\Professional\\Documents\\AliasTrip\\Kruzhok.Pro\\VAIO.KRUZHOK.PRO\\training_6\\cp-0200.ckpt")

img = keras.preprocessing.image.load_img(filepath, target_size=[450, 450])
img = keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255

predict = model.predict_classes(img)

if predict [0][0] == 1:
    print("kruzhok")
