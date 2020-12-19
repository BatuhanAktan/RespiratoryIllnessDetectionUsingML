'''
Using machine learning to aid the diagnosis of COVID-19 in patients.
Author: Batuhan Aktan, Jody Zhou, Sam  ...
Date: DEC 2020
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os
import random
import pickle


data ='./DataSets/chest_xray/train'
category = ['Healthy', 'Pneumonia']
trainingData = []
X = []
y = []


def createTrainingData():
    for element in category:
        path = os.path.join(data, element)
        classNum = category.index(element)
        for file in os.listdir(path):
            imgArray = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            trainingData.append([imgArray, classNum])
    random.shuffle(trainingData)
    return trainingData


def MLPrep(data):
    for arr, cat in data:
        X.append(arr)
        y.append(cat)

    #saves training data X.
    pickleOut = open("X.pickle", "wb")
    pickle.dump(X, pickleOut)
    pickleOut.close()

    #saves training data y
    pickleOut = open("y.pickle", "wb")
    pickle.dump(y, pickleOut)
    pickleOut.close()

MLPrep(createTrainingData())
