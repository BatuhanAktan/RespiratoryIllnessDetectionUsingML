import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


model = keras.models.load_model('MyModel.h5')
categories = ['Healthy', 'Sick']



def prepare(file):
    imgArr = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    imgArr = cv2.resize(imgArr, (600,600))
    
    return np.array(imgArr).reshape(-1, 600,600,1)


prediction = model.predict([prepare('./TestImages/IM-0001-0001.jpeg')])
prediction2 = model.predict([prepare('./TestImages/person15_virus_46.jpeg')])


print('Image being processed:', 'IM-0001-0001.jpeg\nExpected Result: Healthy','\nResult:',categories[int(prediction[0][0])])
print('Image being processed:', 'person15_virus_46.jpeg\nExpected Result: Sick','\nResult:',categories[int(prediction2[0][0])])
