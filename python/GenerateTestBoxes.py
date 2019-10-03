#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 01:45:02 2019

Generate testing boxes for testing out face detection
classifier (i.e. BoxClassifier)

@author: alejandrogonzales
"""

import numpy as np
import cv2
import random
from Rectangle import Rectangle
import copy

# Create the anchor pts hashmap to store all anchors for a frame
def create_anchor_map(colAnchorPts, rowAnchorPts):
    anchorPtsMap = {}
    count = 0
    for m in colAnchorPts:
        for n in rowAnchorPts:
            anchorPtsMap[str(count)] = [m, n]
            count = count + 1
            
    return anchorPtsMap


# Extract pixels given Rectangle object for face/bg
# params:
#   rect - Rectangle object
#   img - Image frame to sample from
def extract_pixels(rect, img, name):

    # loop through rectangles and order the corrds and create cropped image
    oc = rect.order_coords()    # get the ordered coordinates for cropping
    crop = img[oc[0]:oc[1], oc[2]:oc[3]]
    #fname = "object_" + name + ".jpg"
    #cv2.imwrite(fname, crop)
    print("Crop size = " + str(crop.shape))
    print("\n")
    pixels = crop.flatten()
        
    return pixels
        
        
# Create the RPN hashmap using size, scale and aspect ratios
# params:
#   size - rpn default size
#   scale - rpn scale array for adjusting box size
#   ar - rpn aspect ratio for adjust height
def create_rpn_map(size, scale, ar):
    # NOTE: We're making assumption here that AR is always vertical (i.e. 1:2)
    rpn_map = {}
    width = size
    height = size
    
    # Create boxes using width and height arr
    box1 = [size*scale[0], size*scale[0]*ar[0]]
    box2 = [size*scale[0], size*scale[0]*ar[1]]
    box3 = [size*scale[1], size*scale[1]*ar[0]]
    box4 = [size*scale[1], size*scale[1]*ar[1]]
    
    print("Box 1 = " + str(box1))
    print("Box 2 = " + str(box2))
    print("Box 3 = " + str(box3))
    print("Box 4 = " + str(box4))
    
    rpn_map['box1'] = box1
    rpn_map['box2'] = box2
    rpn_map['box3'] = box3
    rpn_map['box4'] = box4
    
    return rpn_map

# Video and data paths for insert rectangular boxes
rootdir = '../../Output/'
videoPath = rootdir + 'videos/';
dataPath = rootdir + 'data/';
croppedPath = rootdir + 'croppedRPN/';

# Prefixes for faces and bg
faces = ['Face_1_', 'Face_2_', 'Face_3_']
bg = ['BG_1_', 'BG_2_', 'BG_3_']

# Suffixes for data and videos
npy = '.npy'
avi = '_rect.avi'

# Region proposal sizes
# NOTE: This was calculated from the below calculate width height max
rpn_size = 120
rpn_scale = [1, 2]
rpn_ar = [1, 2]
rpn_map = create_rpn_map(rpn_size, rpn_scale, rpn_ar)
print("Region Proposal Network Boxes:")
print(str(rpn_map))
print("\n")

# Configuration for face and bg data
# Load an avi video to set the anchor points
numAnchorCols = 10
numAnchorRows = 5

# Frame configuration for resizing and visualizing
step = 30
scale_percent = 50/100

# Video names from the ROIDemo file
videoNames = ['Eating_Back', 'Pointing_Side', 'Writing_Back']

FaceBoxes = []
FrameGroup = []
frameCtr = 0

# Loop through videos and generate test boxes
for i in range(len(videoNames)):
    # Initialize the face and background boxes as dicts
    vidName = videoNames[i]
    
    # Create the video name
    videoName = videoPath + vidName + avi

    # TODO Loop through the videos and data directories to pull the video frames
    # as well as the feature face extraction coords
    cap = cv2.VideoCapture(videoName)

    # Get information of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fp = cap.get(cv2.CAP_PROP_FPS)
        
    # Get the midway frame for each video
    frameNum = frame_count/2
    print("Video frame count = " + str(frame_count))
    print("Frame number = " + str(frameNum))
    print("\n")
    
    # read the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum) 
    frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()    

    if ret == True:

        # Convert color image into grayscale imaged
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = int(gray.shape[0]*scale_percent)
        width = int(gray.shape[1]*scale_percent)
        dim = (width, height)
         
        # resize the frame for viewing
        rimg = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

        # Create the anchors in the frame
        colSpacing = int(width/numAnchorCols)
        rowSpacing = int(height/numAnchorRows)
        
        # Create the coord points for the cols
        colAnchorPts = list(range(colSpacing, width, colSpacing))
        rowAnchorPts = list(range(rowSpacing, height, rowSpacing))
        
        # Create the Anchor pts hashmap for coords for each anchor pt
        anchorPtsMap = create_anchor_map(colAnchorPts, rowAnchorPts)
        
        print("Anchor Pts length = " + str(len(list(anchorPtsMap.keys()))))
        
        # Loop through the anchors map while points exist
        for key in list(anchorPtsMap.keys()):
            # anchor pts for current key
            anchorPts = anchorPtsMap[key]
            xanchor = anchorPts[0]
            yanchor = anchorPts[1]
            
            # Draw the randomly chosen anchor pts
            circleCenter = (xanchor, yanchor)
            circleRadius = 3
            circleColor = (0,255,0)
            
            # Draw the first RPN box in the RPN Map (AR = 1:1, Scale = 1)
            # TODO Loop through all rpn boxes and calculate overlap btwn GT face
            boxname = 'box1'
            rpn_box = rpn_map[boxname]
            
            # TEST: Draw the RPN test box 
            # TODO: This will be chosen from the scale/ratio list
            bw = rpn_box[0]
            bh = rpn_box[1]
            half_w = bw/2
            half_h = bh/2
            
            # Create the spacing from the anchor pt for the box
            h_start = half_h + yanchor
            h_end = yanchor - half_h
            w_start = xanchor - half_w
            w_end = xanchor + half_w
            
            # Create the rectangle points for proposal box
            rpn_x1 = int(w_start)
            rpn_y1 = int(h_start)
            rpn_x2 = int(w_end)
            rpn_y2 = int(h_end)
            rpn_rect = Rectangle(rpn_x1, rpn_y1, rpn_x2, rpn_y2)
            
            # Draw the proposal box over anchors
            rpn_x1 = rpn_rect.x1
            rpn_y1 = rpn_rect.y1
            rpn_x2 = rpn_rect.x2
            rpn_y2 = rpn_rect.y2
            
            # Create the RPN hashmap with box pts and anchors
            #rpnMap['rpn_box'] = rpn_rect
            facePixels = extract_pixels(rpn_rect, rimg, 'face')
            
            # Draw the anchor pts and RPN box on the current frame
            cv2.circle(rimg, circleCenter, circleRadius, circleColor, -1)
            cv2.rectangle(rimg, (rpn_rect.x1, rpn_rect.y1), (rpn_rect.x2, rpn_rect.y2), (0,0,255), 2)
        
            # Append pixels for faces and bg to the boxes lists
            FaceBoxes.append(facePixels)
            FrameGroup.append(frameCtr)
            frameCtr = frameCtr + 1
        
        
        # Show the current frame with RPN boxes and training boxes
        cv2.imshow('Video Frame', rimg)
        cv2.imwrite('test_img_' + str(i) + '.jpg', rimg)
        cv2.waitKey(1)
                        
        input("Press Enter to continue...")
        print("\n")
                                                             
# Release the video processors for CV
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

# Save the face boxes and bg boxes to Numpy files for GridSearchCV use
fout = 'Test_FaceBoxes.npy'
np.save(fout, FaceBoxes)
