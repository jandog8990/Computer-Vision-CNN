import numpy as np
import cv2

# Set labels of objects
object_label_1 = 'Face_1_Eating_Back'
object_label_2 = 'Face_2_Eating_Back'
object_label_3 = 'Face_3_Eating_Back'

# Load the npy cooords for each face in the frame
f1 = np.load('Output/Data/'+ object_label_1 + '.npy')
f2 = np.load('Output/Data/'+ object_label_2 + '.npy')
f3 = np.load('Output/Data/'+ object_label_3 + '.npy')

# Video path
videoPath = 'Output/videos/';

print("F1:")
print(f1)
print("F2:")
print(f2)
print("F3:")
print(f3)

# Load an avi video to set the anchor points
numAnchorCols = 5
numAnchorRows = 5

# TODO Loop through the videos and data directories to pull the video frames
# as well as the feature face extraction coords
videoName = 'Eating_Back_rect.avi'
cap = cv2.VideoCapture(videoPath + videoName)
