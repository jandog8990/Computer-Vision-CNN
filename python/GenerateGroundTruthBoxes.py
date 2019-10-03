#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 01:45:02 2019

Generate Ground truth face boxes using RPN procedure
given in real time object detection paper

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

# Clean the background map by removing replicate keys
# params:
#   faceMap - face dict
#   backMap - backgorund dict
def clean_back_map(faceMap, backMap):
    for fname, fdata in faceMap.items():
        for fkey in fdata.keys():
            if key in backMap:
                del backMap[fkey]   
            
    return backMap

# Extract pixels given Rectangle object for face/bg
# params:
#   rectList - list of Rectangle objects
#   img - Image frame to sample from
def extract_pixels(rectList, img, name):

    # loop through rectangles and order the corrds and create cropped image
    pixelList = []
    count = 0
    for rect in rectList:
        oc = rect.order_coords()    # get the ordered coordinates for cropping
        crop = img[oc[0]:oc[1], oc[2]:oc[3]]
        fname = "object_" + name + "_" + str(count) + ".jpg"
        count = count + 1
        cv2.imwrite(fname, crop)
        pixelList.append(crop.flatten())
        
    return pixelList

# Find the max area for intersection face boxes
# params:
#   faceMap - face dict
def find_max_face(faceMap):
    rpnMap = {}
    
    for fkey,fdata in faceMap.items():
        
        # initialize the rpn map
        rpnMap = copy.deepcopy(fdata)
        faceMap[fkey] = {}

        # initialize max areas and counts
        maxArea = 0
        maxKey = ''
        count = 0
        while (len(rpnMap) > 1):
            rkey = random.sample(list(rpnMap.keys()), 1)[0]
            rdata = rpnMap[rkey]
            
            # delete all items that are below the max
            if (count == 0):
                maxArea = rdata['intersect_area']
                maxKey = rkey
            else:
                if (rdata['intersect_area'] >= maxArea):
                    
                    # set new max area and delete the previous max from map
                    maxArea = rdata['intersect_area']
                    rpn_rect = rdata['rpn_box']
                    rpnArea = rpn_rect.area()
                    faceArea = rdata['f_area']
                    IoU = maxArea/(rpnArea + faceArea)
                    
                    maxKey = rkey
                else:
                    # delete the minimum element in rpn map
                    del rpnMap[rkey]
            count= count +1
        
        # update the faceMap to include RPN box for the max area
        key = list(rpnMap.keys())[0]
        faceMap[fkey] = rpnMap[key]['rpn_box']
        
    return faceMap
        

# Randomly sample the background map for the number of boxes
# params:
#   backMap - background hashmap
#   numBoxes - number of object boxes
def sample_back_map(backMap, numBoxes):
    keys = list(backMap.keys())
    newBackMap = {}
    for i in range(numBoxes):
        randKey = random.choice(keys)
        newBackMap[randKey] = backMap[randKey]
        
    return newBackMap
        
        
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

# Calculate the area of face box using coords and width/height
# params:
#   roi - region of interest (x, y, w, h)
#   scale - scale of the width height
def calculate_wh(roi, scale):
    w = roi[2]
    h = roi[3]
    
    width = w*scale
    height = h*scale

    return [width, height]

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
#numAnchorRows = 5   # box1 rows
numAnchorRows = 3   # box2 rows

# Frame configuration for resizing and visualizing
step = 30
scale_percent = 50/100

# Video names from the ROIDemo file
#videoNames = {}
#videoNames['back'] = ['Eating1Round2', 'Pointing1Round2', 'Talking2Round1', 'Typing3Round1', 'Writing1Round2']
#videoNames['side'] = ['Eating1Round2Side', 'Pointing2Round2Side', 'Talking3Round1Side', 'Typing2Round1Side', 'Writing1Round2Side']

# Label names from ROI
labelNames = {}
labelNames['back'] = ['Eating_Back', 'Pointing_Back', 'Talking_Back', 'Typing_Back', 'Writing_Back']
labelNames['side'] = ['Eating_Side', 'Pointing_Side', 'Talking_Side', 'Typing_Side', 'Writing_Side']

print("Data and Videos:")
FaceBoxes = []
BackBoxes = []
FrameGroup = []
frameCtr = 0
#FaceBoxes = {}
#BackBoxes = {}
for key in labelNames.keys():
    print("key = " + key)
    labels = labelNames[key]
    
    # Loop through labels and create the face, bg, and video data
    for i in range(len(labels)):
        # Initialize the face and background boxes as dicts
        vidName = labels[i]
        #FaceBoxes[vidName] = {}
        #BackBoxes[vidName] = {}
        
        # Create the npy data file names
        face1_data = dataPath + faces[0] + vidName + npy
        face2_data = dataPath + faces[1] + vidName + npy
        face3_data = dataPath + faces[2] + vidName + npy
        bg1_data = dataPath + bg[0] + vidName + npy
        bg2_data = dataPath + bg[1] + vidName + npy
        bg3_data = dataPath + bg[2] + vidName + npy
        
        # Create the video name
        videoName = videoPath + vidName + avi
        
        print("Face 1 data = " + str(face1_data))
        print("BG 1 data = " + str(bg1_data))
        print("Video name = " + str(videoName))
        print("------------------------------------------\n")

        # Load the npy cooords for each face in the frame (X, Y, W, H)
        f1 = np.load(face1_data)
        f2 = np.load(face2_data)
        f3 = np.load(face3_data)
        
        # Scale the face data and create hashmap of faces
        f1 = [int(coord*scale_percent) for coord in f1]
        f2 = [int(coord*scale_percent) for coord in f2]
        f3 = [int(coord*scale_percent) for coord in f3]
        face_data = {'face1': f1, 'face2': f2, 'face3': f3}
        NUM_OBJECT_BOXES = int(len(face_data))
        
        # Calculate face areas
        [w1, h1] = calculate_wh(f1, scale_percent)
        [w2, h2] = calculate_wh(f2, scale_percent)
        [w3, h3] = calculate_wh(f3, scale_percent)
        
        # Create the proposal boxes using the max sizes of the faces
        max_width = max(w1, w2, w3)
        max_height = max(h1, h2, h3)
        
        print("Face Data:")
        print("F1: " + str(f1))
        print("F2: " + str(f2))
        print("F3: " + str(f3))
        print("\n")
        
        print("Max Width/Height Proposals = " + str(max_width) + " / " + str(max_height))
        print("\n")

        # TODO Loop through the videos and data directories to pull the video frames
        # as well as the feature face extraction coords
        cap = cv2.VideoCapture(videoName)

        # Get information of the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        fp = cap.get(cv2.CAP_PROP_FPS)
        
        frame_arr = [x for x in range(0, frame_count, step)]
        print("Video frame count = " + str(frame_count))
        print("Step frame arr size = " + str(len(frame_arr)))
        print("\n")

        # Loop through the video and pull out every 10th frame
        for i in range(0, frame_count, step):
            # set the frame number
            frameNum = i
        
            # read the current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
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
                        
                # Initialize the Face and BG hashmaps
                faceMap = {}
                for fkey in face_data.keys():
                    faceMap[fkey] = {}
                backMap = {}
                
                # Loop through the anchors map while points exist
                while (len(anchorPtsMap) > 0):
                    # Randomly select an anchor pt in the map
                    randKey = random.choice(list(anchorPtsMap.keys()))
                    anchorPts = anchorPtsMap[randKey]
                    
                    # Remove the anchor point that was just chosen
                    del anchorPtsMap[randKey]
                    xanchor = anchorPts[0]
                    yanchor = anchorPts[1]
                    
                    # Draw the randomly chosen anchor pts
                    circleCenter = (xanchor, yanchor)
                    circleRadius = 3
                    circleColor = (0,255,0)
                    
                    # TEST: Draw the first box in the RPN map
                    # TODO Loop through all rpn boxes and calculate overlap btwn GT face
                    boxname = 'box2'
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
                    
                    # Loop through faces & calculate intersections with face boxes
                    rpnMap = {}
                    for fname, fdata in face_data.items():
                        
                        x = fdata[0]
                        y = fdata[1]
                        w = fdata[2]
                        h = fdata[3]
                        x11 = x
                        y11 = y
                        x22 = x+w
                        y22 = y+h
                        frect = Rectangle(x11, y11, x22, y22)
                        farea = frect.area()
                                                    
                        intersection = rpn_rect&frect
                        if (intersection != None):
                            # Area of intersection
                            intersection_area = intersection.area()
                            
                            # Draw the proposal box over anchors
                            rpn_x1 = rpn_rect.x1
                            rpn_y1 = rpn_rect.y1
                            rpn_x2 = rpn_rect.x2
                            rpn_y2 = rpn_rect.y2
                            
                            # Create the RPN hashmap with box pts and intersection areas
                            rpnMap['rpn_box'] = rpn_rect
                            rpnMap['anchor_pt'] = (xanchor, yanchor)
                            rpnMap['intersect_area'] = intersection_area
                            rpnMap['f_area'] = farea
                            
                            # Append the RPN hashmap to the Face hashmap
                            faceMap[fname][randKey] = rpnMap  
                    
                            # Draw the anchor pts and RPN box on the current frame
                            cv2.circle(rimg, circleCenter, circleRadius, circleColor, -1)
                            cv2.rectangle(rimg, (rpn_rect.x1, rpn_rect.y1), (rpn_rect.x2, rpn_rect.y2), (0,0,255), 2)
                        else:
                            # Append the non-intersecting box to the backMap
                            if (randKey != None):
                                backMap[randKey] = rpn_rect
                
                # Clean the background hash map and select random background box
                backMap = clean_back_map(faceMap, backMap)
                backMap = sample_back_map(backMap, NUM_OBJECT_BOXES)
                
                # Outside Anchor Loop: Find the face box with the max area
                faceMap = find_max_face(faceMap)
                
                # Convert hashmap values to list of rectangles
                faceList = list(faceMap.values())
                backList = list(backMap.values())
                
                # Extract pixels from faces and backgrounds for current frame
                facePixelsList = extract_pixels(faceList, rimg, 'face')
                backPixelsList = extract_pixels(backList, rimg, 'back')
                
                # Append pixels for faces and bg to the boxes lists
                FaceBoxes.extend(facePixelsList)
                BackBoxes.extend(backPixelsList)
                FrameGroup.append(frameCtr)
                frameCtr = frameCtr + 1
                
                # Show the current frame with RPN boxes and training boxes
                cv2.imshow('Video Frame', rimg)
                cv2.waitKey(1)
                                
                input("Press Enter to continue...")
                print("\n")
                                                             
# Release the video processors for CV
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

# Save the face boxes and bg boxes to Numpy files for GridSearchCV use
fout = 'FaceBoxes.npy'
bout = 'BackBoxes.npy'
np.save(fout, FaceBoxes)
np.save(bout, BackBoxes)
