'''
Using machine learning to aid the diagnosis of COVID-19 in patients.
Author: Batuhan Aktan, Jody Zhou, Sam  ...
Date: DEC 2020
'''
import os
import random
import pickle
import cv2
import numpy as np

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
            imgArray = cv2.resize(imgArray, (600,600))
            trainingData.append([imgArray, classNum])
    random.shuffle(trainingData)
    return trainingData


def MLPrep(data):
    global X
    global y
    for arr, cat in data:
        X.append(arr)
        y.append(cat)

        
    X = np.array(X).reshape(-1,600,600,1)
    y = np.array(y)

    
    #saves training data X.
    pickleOut = open("X.pickle", "wb")
    pickle.dump(X, pickleOut)
    pickleOut.close()

    #saves training data y
    pickleOut = open("y.pickle", "wb")
    pickle.dump(y, pickleOut)
    pickleOut.close()

MLPrep(createTrainingData())
