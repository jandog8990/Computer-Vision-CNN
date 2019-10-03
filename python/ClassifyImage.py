#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 01:45:02 2019

Classify image class that takes in a single image
and uses the max classifier to train and test

@author: alejandrogonzales
"""
import numpy as np
from PIL import Image
    
"""
Generates boxes for a single image and a single classifier.
"""
def classify_image(boxClassifier, FaceBoxes, rpn_size):
    # Use the final classifier to predict the FaceBoxes
    # NOTE: This depends on running FaceDetection_FinalTrain which gets the best params
    boxPred = boxClassifier.predict(FaceBoxes)
    ix = np.where(boxPred == 1)[0]
    print("Box Prediction:")
    print(boxPred)
    print("\n")
    print("True Indices:")
    print(ix)
    print("\n")
        
    # Loop through indices and plot the images
    for idx in ix:
        ibox = np.reshape(FaceBoxes[idx], (rpn_size, rpn_size))
        img_box = Image.fromarray(ibox)
        img_box.save('face_pred_box_' + str(idx) + '.jpg')
        img_box.show()

# Import FaceBoxes from the GenerateTestBoxes script
# which creates test face box pixels
dataDir = '../python/'
FaceBoxes = np.load(dataDir+'Test_FaceBoxes.npy', allow_pickle=True)

# Classify image face boxes
rpn_size = 120
classify_image(clf, FaceBoxes, rpn_size)
