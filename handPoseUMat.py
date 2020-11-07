import cv2
import time
import numpy as np
import math

from UMatFileVideoStream import UMatFileVideoStream

from piano import Piano

piano = Piano()

def playMusic(x, y):
    print('point:', x, y)
    key = piano.press(x, y)
    if key is not None:
      print(key+1)

def keyPressed():
    for finger in range(5):
        x1 = points[0, finger][0]
        y1 = points[0, finger][1]
        if x1 == 0 and y1 == 0:
            continue
        x2 = points[1, finger][0]
        y2 = points[1, finger][1]
        if x2 == 0 and y2 == 0:
            continue

        # x1, y2 is the current value
        # downward means distance travelled have to be +ve if end - start
        distance = math.sqrt(pow((x2-x1),2) + pow((y2-y1), 2))
        if distance > press_thres:
            playMusic(x2, y2)

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

# parameters
threshold = 0.2
press_thres = 8
frame_gap = 5

input_source = "input.mp4"
# cap = cv2.VideoCapture(0)
# hasFrame, frame = cap.read()

# umat
cap = UMatFileVideoStream(input_source)
frameWidth = cap.width
frameHeight = cap.height
aspect_ratio = frameWidth/frameHeight

# try:
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]

#     aspect_ratio = frameWidth/frameHeight
# except:
#     aspect_ratio = 1

inHeight = 300
inWidth = int(((aspect_ratio*inHeight)*8)//8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = cv2.VideoWriter('output_UMat.mp4',fourcc, 15, (frameWidth,frameHeight))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# setting GPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cv2.ocl.setUseOpenCL(True)

# matrix points of size 2x5 (2 sets of data, for 5 fingers each)
points = np.zeros((2, 5), dtype=(int,2))
frame_counter = 0

while 1:
    t = time.time()

    # hasFrame, frame = cap.read()
    frame = cap.read()
    # frameCopy = np.copy(frame)
    # frameCopy = frame
    
    try:
        frame = frame.getMat(cv2.ACCESS_READ)
        piano.display(frame, frameHeight, frameWidth)
        frame = cv2.flip(frame, 1)
        cv2.imshow("Piano", frame)
    except:
        print('end of video')
        break

    # if not hasFrame:
    #     cv2.waitKey()
    #     break

    if not frame:
        cv2.waitKey()
        break

    # resized = cv2.resize(frame, (500,300))
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)
    output = net.forward()
    # print("Time taken for net = {}".format(time.time() - t))

    # Empty list to store the detected keypoints
    skip = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    for i in range(nPoints):
        if i in skip: continue
        finger = int(i/4 - 1)
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if frame_counter%frame_gap == 0 and finger == 0:
            points[0, :] = points[1, :]

        # if frame number is a multiple of 5
        if frame_counter%frame_gap == 0:
            if prob > threshold:
                cv2.circle(frame, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                points[1, finger] = (int(point[0]), int(point[1]))
                # cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            else :
                points[1, finger] = (0, 0)

        if frame_counter%frame_gap == 0 and finger == 4:
            # print(points)
            keyPressed()

    # cv2.imshow('Output-Skeleton', frame)
    # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    frame_counter += 1
    time_taken = time.time() - t
    fps = 1.0 / time_taken
    # print("fps= {}".format(fps))
    
    vid_writer.write(frame)

vid_writer.release()