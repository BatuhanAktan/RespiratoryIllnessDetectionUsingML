import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


model = keras.models.load_model('MyModel.h5')
categories = ['Healthy', 'Sick']



def prepare(file):
    imgArr = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    newArr = cv2.resize(imgArr, (600,600))
    
    return np.array(newArr).reshape(-1, 600,600,1)


prediction = model.predict([prepare('IM-0166-0001.jpeg')])


print(categories[int(prediction[0][0])])
