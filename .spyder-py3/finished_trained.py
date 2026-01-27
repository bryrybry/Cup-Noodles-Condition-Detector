# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:07:49 2026

@author: eizin
"""

import tensorflow as tf
import numpy as np

MODEL_PATH = r"cupnoodles_model.keras"
IMG_PATH = r"C:\Users\eizin\.spyder-py3\rps\CupNoodles\predict\IMG_4299.jpg"  # change this

IMG_SIZE = (224, 224)
class_names = ['Finished', 'Opened', 'Sealed']  # EXACT order from training output

model = tf.keras.models.load_model(MODEL_PATH)

img = tf.keras.utils.load_img(IMG_PATH, target_size=IMG_SIZE)
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

pred = model.predict(x)[0]
idx = np.argmax(pred)

print("Predicted:", class_names[idx])
for name, p in zip(class_names, pred):
    print(f"{name}: {p:.3f}")
