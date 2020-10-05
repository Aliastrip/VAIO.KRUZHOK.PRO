from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
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
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(256, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    return model


train_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
)
test_datagen = image.ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    "files",
    target_size=(450, 450),
    batch_size=10,
    class_mode='binary'
)

valid_gen = test_datagen.flow_from_directory(
    "test",
    target_size=(450, 450),
    batch_size=10,
    class_mode='binary'
)

model = build_model()

checkpoint_path = "training_4/cp-{epoch:04d}.ckpt" # поменять папку +1
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=100) # поменять на 100

his = model.fit_generator(
    train_gen,
    steps_per_epoch=10,
    epochs=1000, # поменять на 1000
    callbacks=[cp_callback],
    validation_data=valid_gen,
    validation_steps=1
)

model.save("test3.h5") # поменять модель на +1

acc = his.history['acc']
val_acc = his.history['val_acc']

loss = his.history['loss']
val_loss = his.history['val_loss']

plt.plot(acc)
plt.plot(val_acc)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
