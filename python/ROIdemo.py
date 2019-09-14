# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:25:15 2019

@author: wenjing
@editor: Jandog Productions
"""

import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# videoNames = ['Eating1Round2', 'Pointing1Round2', 'Talking2Round1', 'Typing3Round1', 'Writing1Round2', 
videoNames = ['Eating1Round2Side', 'Pointing2Round2Side', 'Talking3Round1Side', 'Typing2Round1Side', 'Writing1Round2Side']
# labelNames = ['Eating_Back', 'Pointing_Back', 'Talking_Back', 'Typing_Back', 'Writing_Back', 
labelNames = ['Eating_Side', 'Pointing_Side', 'Talking_Side', 'Typing_Side', 'Writing_Side']

for i in range(len(videoNames)):
    videoName = videoNames[i]
    labelName = labelNames[i]
    print("Video name = " + videoName)
    print("Label name = " + labelName)
    print("\n")
    
    # Face labels for objects
    face_label_1 = 'Face_1_'
    face_label_2 = 'Face_2_'
    face_label_3 = 'Face_3_'

    # BG labels for objects
    bg_label_1 = 'BG_1_'
    bg_label_2 = 'BG_2_'
    bg_label_3 = 'BG_3_'

    cap = cv2.VideoCapture('SideCameraClips/' + videoName + '.mp4')

    # Face label directions for faces
    face_label_1 = face_label_1 + labelName
    face_label_2 = face_label_2 + labelName
    face_label_3 = face_label_3 + labelName
    
    # BG label directions for faces
    bg_label_1 = bg_label_1 + labelName
    bg_label_2 = bg_label_2 + labelName
    bg_label_3 = bg_label_3 + labelName

    # Get information of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fp = cap.get(cv2.CAP_PROP_FPS)

    print("Frame count = " + str(frame_count));
    print("Height / width = " + str(height) + " / " + str(width));
    print("Frames per/sec = " + str(fp));

    # Read the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    ret, frame_1 = cap.read()

    # Convert color image into grayscale image
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

    # Save the first frame
    cv2.imwrite('Output/FirstFrame/' + videoName +'.jpg', gray_1)

    # Select ROI
    fromCenter = False 
    ROIs = cv2.selectROIs('Select ROIs', frame_1, fromCenter) 
    #r = cv2.selectROI(frame_1)
    #r2 = cv2.selectROI(frame_1)
    #print('r2 = ', r2)
    print('ROIs =', ROIs)
    
    # Face ROIs	
    r1 = ROIs[0];
    r2 = ROIs[1];
    r3 = ROIs[2];
    
    # Background ROIs
    b1 = ROIs[3];
    b2 = ROIs[4];
    b3 = ROIs[5];

    # Number of elements in ROI
    numrows = len(ROIs)
    numcols = len(ROIs[0])
    print('ROI Size:')
    print('Num rows = ' + str(numrows))
    print('Num cols = ' + str(numcols))

    # Get the different ROI components
    #ROI_1 = frame_1[ROIs[0][1]:ROIs[0][1]+ROIs[0][3], ROIs[0][0]:ROIs[0][0]+ROIs[0][2]]
    #ROI_2 = frame_1[ROIs[1][1]:ROIs[1][1]+ROIs[1][3], ROIs[1][0]:ROIs[1][0]+ROIs[1][2]]
    #ROI_3 = frame_1[ROIs[2][1]:ROIs[2][1]+ROIs[2][3], ROIs[2][0]:ROIs[2][0]+ROIs[2][2]]

    # Save the coordinate and size of the face rectangles
    np.save('Output/Data/'+ face_label_1 + '.npy', r1)    
    np.save('Output/Data/'+ face_label_2 + '.npy', r2)   
    np.save('Output/Data/'+ face_label_3 + '.npy', r3)
    
    # Save the coordinate and size of the background rectangles
    np.save('Output/Data/'+ bg_label_1 + '.npy', b1) 
    np.save('Output/Data/'+ bg_label_2 + '.npy', b2)   
    np.save('Output/Data/'+ bg_label_3 + '.npy', b3)

    # Define the codec and create VideoWriter objects

    # Full sized frame video
    outRect = cv2.VideoWriter('Output/Videos/' + labelName + '_rect.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (width, height), 0)

    # Cropped face videos (Face 1-3)
    outCropped1 = cv2.VideoWriter('Output/Videos/' + face_label_1 + '_cropped_1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (r1[2], r1[3]), 0) 
    outCropped2 = cv2.VideoWriter('Output/Videos/' + face_label_2 + '_cropped_2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (r2[2], r2[3]), 0) 
    outCropped3 = cv2.VideoWriter('Output/Videos/' + face_label_3 + '_cropped_3.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (r3[2], r3[3]), 0)
    
    # Cropped bg videos (BG 1-3)
    bgCropped1 = cv2.VideoWriter('Output/Videos/' + bg_label_1 + '_cropped_1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (b1[2], b1[3]), 0) 
    bgCropped2 = cv2.VideoWriter('Output/Videos/' + bg_label_2 + '_cropped_2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (b2[2], b2[3]), 0) 
    bgCropped3 = cv2.VideoWriter('Output/Videos/' + bg_label_3 + '_cropped_3.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fp, (b3[2], b3[3]), 0)

    cv2.destroyAllWindows()  

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            # Convert color image into grayscale image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Crop the face frames 
            imCrop1 = gray.copy()[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]
            imCrop2 = gray.copy()[int(r2[1]):int(r2[1]+r2[3]), int(r2[0]):int(r2[0]+r2[2])]
            imCrop3 = gray.copy()[int(r3[1]):int(r3[1]+r3[3]), int(r3[0]):int(r3[0]+r3[2])]

            # Draw rectangle around current face frame
            cv2.rectangle(gray, (int(r1[0]), int(r1[1])), (int(r1[0]+r1[2]), int(r1[1]+r1[3])), (0,0,255), 3)
            cv2.rectangle(gray, (int(r2[0]), int(r2[1])), (int(r2[0]+r2[2]), int(r2[1]+r2[3])), (0,0,255), 3)
            cv2.rectangle(gray, (int(r3[0]), int(r3[1])), (int(r3[0]+r3[2]), int(r3[1]+r3[3])), (0,0,255), 3)
            
            # Crop the bg frames 
            bgCrop1 = gray.copy()[int(b1[1]):int(b1[1]+b1[3]), int(b1[0]):int(b1[0]+b1[2])]
            bgCrop2 = gray.copy()[int(b2[1]):int(b2[1]+b2[3]), int(b2[0]):int(b2[0]+b2[2])]
            bgCrop3 = gray.copy()[int(b3[1]):int(b3[1]+b3[3]), int(b3[0]):int(b3[0]+b3[2])]

            # Draw rectangle around current face frame
            cv2.rectangle(gray, (int(b1[0]), int(b1[1])), (int(b1[0]+b1[2]), int(b1[1]+b1[3])), (0,0,255), 3)
            cv2.rectangle(gray, (int(b2[0]), int(b2[1])), (int(b2[0]+b2[2]), int(b2[1]+b2[3])), (0,0,255), 3)
            cv2.rectangle(gray, (int(b3[0]), int(b3[1])), (int(b3[0]+b3[2]), int(b3[1]+b3[3])), (0,0,255), 3)

            # write frames
            outRect.write(gray)
            outCropped1.write(imCrop1)
            outCropped2.write(imCrop2)
            outCropped3.write(imCrop3)
            bgCropped1.write(bgCrop1)
            bgCropped2.write(bgCrop2)
            bgCropped3.write(bgCrop3)

            cv2.imshow('Grayscale Video w Rectangle', gray)
            #cv2.imshow('With Rectangle', frame)
#             cv2.imshow('Cropped Face 1', imCrop1)
#             cv2.imshow('Cropped Face 2', imCrop2)
#             cv2.imshow('Cropped Face 3', imCrop3)

        # Press 'q' to close all windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release everything if job is finished            
    cap.release()
    outRect.release()
    outCropped1.release()
    outCropped2.release()
    outCropped3.release()
    bgCropped1.release()
    bgCropped2.release()
    bgCropped3.release()
    cv2.destroyAllWindows()

