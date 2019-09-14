import numpy as np
import cv2

# Video names from the ROIDemo file
videoNames = {}
videoNames['back'] = ['Eating1Round2', 'Pointing1Round2', 'Talking2Round1', 'Typing3Round1', 'Writing1Round2']
videoNames['side'] = ['Eating1Round2Side', 'Pointing2Round2Side', 'Talking3Round1Side', 'Typing2Round1Side', 'Writing1Round2Side']

# Label names from ROI
labelNames = {}
labelNames['back'] = ['Eating_Back', 'Pointing_Back', 'Talking_Back', 'Typing_Back', 'Writing_Back']
labelNames['side'] = ['Eating_Side', 'Pointing_Side', 'Talking_Side', 'Typing_Side', 'Writing_Side']


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
numAnchorCols = 10
numAnchorRows = 5

# TODO Loop through the videos and data directories to pull the video frames
# as well as the feature face extraction coords
videoName = 'Eating_Back_rect.avi'
cap = cv2.VideoCapture(videoPath + videoName)

# Get information of the video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fp = cap.get(cv2.CAP_PROP_FPS)

# Frame configuration for resizing and visualizing
step = 20
scale_percent = 50/100

print("Frame count = " + str(frame_count));
print("Height / width = " + str(height) + " / " + str(width));
print("Frames per/sec = " + str(fp));

# Loop through the video and pull out every 10th frame
# while(cap.isOpened()):
for i in range(0, frame_count, step):
#     print("Reading frame [" + str(i) + "]")

    # read the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
#     print("Capture current frame:")
    frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    ret, frame = cap.read()    

    if ret == True:

        # Convert color image into grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = int(gray.shape[0]*scale_percent)
        width = int(gray.shape[1]*scale_percent)
        dim = (width, height)
        print("Frame [ " + str(frame_pos) + " ], (" + str(height) + ", " + str(width) + ")")
        

        # Create the anchors in the frame
        colSpacing = int(width/numAnchorCols)
        rowSpacing = int(height/numAnchorRows)
        print("Row spacing = " + str(colSpacing))
        print("Col spacing = " + str(rowSpacing))
        
        # Create the coord points for the cols
        colAnchorPts = list(range(colSpacing, width, colSpacing))
        rowAnchorPts = list(range(rowSpacing, height, rowSpacing))
        print("Column anchor pts:")
        print(colAnchorPts)
        print("Row anchor pts:")
        print(rowAnchorPts)
        print("\n")
        
                
        # resize the frame for viewing
        rimg = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
        
        # loop through anchor pts and insert small circles
        for m in range(len(rowAnchorPts)):
            for n in range(len(colAnchorPts)):
                circleCenter = (colAnchorPts[n], rowAnchorPts[m])
                circleRadius = 3
                circleColor = (0,255,0)
                cv2.circle(rimg, circleCenter, circleRadius, circleColor, -1)

        cv2.imshow('Video Frame', rimg)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
        
    #cv2.waitKey(1)
#     else: 
#         break;

# Press 'q' to close all windows
# & 0xFF == ord('q'):
#     break
        
# Press 'q' to close all windows
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break          
cap.release()
# cv2.waitKey(1) & 0xFF == ord('q')
cv2.destroyAllWindows()
cv2.waitKey(1)