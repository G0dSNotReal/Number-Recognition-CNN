import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

dataset_folder = 'D:/User/Code/Fish-Freshness-Detection-master/Uji'
model_path = 'D:/User/Code/Fish-Freshness-Detection-master/modelCNN.h5'

model = load_model(model_path)

for filename in os.listdir(dataset_folder):
    img_path = os.path.join(dataset_folder, filename)
    img = tf.keras.utils.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    print(f"Image: {filename}, Predicted Label: {predicted_label}")