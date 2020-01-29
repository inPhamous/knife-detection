import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle

DATADIR = "/Users/johnpham/Desktop/knife-detection"
CATEGORIES = ["knife"]
IMG_SIZE = 200

trainingData = []

def createTrainingData():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        classNum = CATEGORIES.index(category)
        for img in os.listdir(path):
            try: 
                imgArray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resizedArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))
                trainingData.append([resizedArray, classNum])
            except Exception as e:
                pass

createTrainingData()

X = [] #features
Y = [] #label 0 == knife

for features, label in trainingData:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickleOut = open("X.pickle", "wb")
pickle.dump(X, pickleOut)
pickleOut.close()

pickleOut = open("Y.pickle", "wb")
pickle.dump(Y, pickleOut)
pickleOut.close()

"""
pickleIn = open("X.pickle", "rb")
X = pickle.load(pickleIn)

print(X[1])
"""
