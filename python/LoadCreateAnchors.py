import numpy as np
import cv2
from Rectangle import Rectangle

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
step = 20
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
for key in labelNames.keys():
    print("key = " + key)
    labels = labelNames[key]
    
    # Loop through labels and create the face, bg, and video data
    for i in range(len(labels)):
        # Create the npy data file names
        face1_data = dataPath + faces[0] + labels[i] + npy
        face2_data = dataPath + faces[1] + labels[i] + npy
        face3_data = dataPath + faces[2] + labels[i] + npy
        bg1_data = dataPath + bg[0] + labels[i] + npy
        bg2_data = dataPath + bg[1] + labels[i] + npy
        bg3_data = dataPath + bg[2] + labels[i] + npy
        
        # Create the video name
        videoName = videoPath + labels[i] + avi
        
        print("Face 1 data = " + str(face1_data))
        print("BG 1 data = " + str(bg1_data))
        print("Video name = " + str(videoName))
        print("------------------------------------------\n")

        # Load the npy cooords for each face in the frame (X, Y, W, H)
        f1 = np.load(face1_data)
        f2 = np.load(face2_data)
        f3 = np.load(face3_data)
        face_data = {'face1': f1, 'face2': f2, 'face3': f3}
        
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
        
        print("Face Width Heigh Scale:")
        print("Face 1 = " + str(w1) + " : " + str(h1))
        print("Face 2 = " + str(w2) + " : " + str(h2))
        print("Face 3 = " + str(w3) + " : " + str(h3))
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

        # Loop through the video and pull out every 10th frame
        # while(cap.isOpened()):
        for i in range(0, frame_count, step):
        
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
        
                # Create the anchors in the frame
                colSpacing = int(width/numAnchorCols)
                rowSpacing = int(height/numAnchorRows)
                
                # Create the coord points for the cols
                colAnchorPts = list(range(colSpacing, width, colSpacing))
                rowAnchorPts = list(range(rowSpacing, height, rowSpacing))
                
                # resize the frame for viewing
                rimg = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
                
                # TEST: Draw the first box in the RPN map
                # TODO Loop through all rpn boxes and calculate overlap btwn GT face
                boxname = 'box2'
                rpn_box = rpn_map[boxname]
                
                # TEST: Draw the RPN test box
                bw = rpn_box[0]
                bh = rpn_box[1]
                half_w = bw/2
                half_h = bh/2
                
                # loop through anchor pts and insert small circles
                # TODO: Randomly sample the anchor pts rather than looping
                for m in range(0, len(rowAnchorPts), 2):  # testing row stride
                    for n in range(0, len(colAnchorPts), 2):    # testing col stride
                        xanchor = colAnchorPts[n]
                        yanchor = rowAnchorPts[m]
                                
                        circleCenter = (xanchor, yanchor)
                        circleRadius = 3
                        circleColor = (0,255,0)
                        cv2.circle(rimg, circleCenter, circleRadius, circleColor, -1)
                        
                        # Loop through proposals and create Rectangle objects
                        # TODO: Cr
                        h_start = half_h + yanchor
                        h_end = yanchor - half_h
                        w_start = xanchor - half_w
                        w_end = xanchor + half_w
                        
                        # Create the rectangle points for proposal
                        x1 = int(w_start)
                        y1 = int(h_start)
                        x2 = int(w_end)
                        y2 = int(h_end)
                        rpn_rect = Rectangle(x1, y1, x2, y2)
                        
                        # Calculate intersections with face boxes
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
                                                        
                            intersection = rpn_rect&frect
                            if (intersection != None):
                                
                                # Draw the proposal box over anchors
                                rpn_x1 = rpn_rect.x1
                                rpn_y1 = rpn_rect.y1
                                rpn_x2 = rpn_rect.x2
                                rpn_y2 = rpn_rect.y2
                                
                                print("Frame position = " + str(frame_pos))
                                print("Face name = " + str(fname))
                                print("Box name = " + str(boxname))
                                print("Anchor [ " + str(xanchor) + ", " + str(yanchor) + " ]")
                                print("-----------------------------------------")
                                print("RPN Box location = " + rpn_rect.to_string())
                                print("Face location = " + frect.to_string())
                                print("intersection = " + intersection.to_string())
                                print("-----------------------------------------\n")
                                
                                # Only draw the rpn boxif intersection exists
                                cv2.rectangle(rimg, (rpn_x1,rpn_y1), (rpn_x2,rpn_y2), (0,0,255), 2)

        
                cv2.imshow('Video Frame', rimg)
        
        #        if cv2.waitKey(1) & 0xFF == ord('q'):
        #            break       
                    
                cv2.waitKey(1)
                
        input("Press Enter to continue...")

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