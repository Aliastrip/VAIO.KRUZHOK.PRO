from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from tensorflow.keras.preprocessing import image
# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import os, shutil

filepath = input("Set path")
model = keras.models.load_model("test3.h5")
data = []
i = 0
for (dir_path, _, filenames) in os.walk(filepath):
    for file in filenames:
        img = keras.preprocessing.image.load_img(dir_path + "\\" + file, target_size=[450, 450])
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255
        i += 1
        print(dir_path + "\\" + file)
        predict = model.predict_classes(img)
        data.append(predict[0][0])

print(sum(data)/i)