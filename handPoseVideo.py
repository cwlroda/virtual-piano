import cv2
import time
import numpy as np

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22

threshold = 0.2

input_source = "ezgif.com-gif-maker (2).mp4"
cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = int(360)
inWidth = int(((aspect_ratio*inHeight)*8)//8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = cv2.VideoWriter('output_tmp.mp4',fourcc, 30, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame_counter = 0
frame_gap = 10
points = np.zeros(5, dtype=(int,2))
old_points = np.zeros(5, dtype=(int,2))
num = None

while 1:
    t = time.time()
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        cv2.waitKey()
        break

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (int(inWidth), int(inHeight)),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    fingertips = [4, 8, 12, 16, 20]

    for i in fingertips:
        finger = int(i/4 - 1)
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # if frame number is a multiple of 5
        if prob > threshold:
            cv2.circle(frame, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            points[finger] = (int(point[0]), int(point[1]))    
            
            if frame_counter % frame_gap == 0:
                old_points[finger] = (int(point[0]), int(point[1]))
            # cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            if int(point[1]) - old_points[finger][1] > 50:
                num = finger + 1
    
    cv2.putText(frame, "Finger {} pressed".format(num), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Skeleton', frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    frame_counter += 1
    print("Total time = {}".format(time.time() - t))

    vid_writer.write(frame)

vid_writer.release()

