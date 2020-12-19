'''
Using machine learning to aid the diagnosis of COVID-19 in patients.
Author: Batuhan Aktan, Jody Zhou, Sam  ...
Date: DEC 2020
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

file ='./DataSets/HealthyLungs/Healthy.jpg'
img = cv2.imread(file)
cv2.imshow('Image', img)
cv2.waitkey(0)

input()
