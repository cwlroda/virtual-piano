import cv2
import time

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22

threshold = 0.001

input_source = "input.mp4"
cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = int(240)
inWidth = int(((aspect_ratio*inHeight)*8)//8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = cv2.VideoWriter('output.mp4',fourcc, 30, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

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

    # Empty list to store the detected keypoints
    skip = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    for i in range(nPoints):
        if i in skip: continue
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :   
            print(prob)         
            print(point)
            cv2.circle(frame, (int(point[0]), int(point[1])), 6, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    print("Time Taken for frame = {}".format(time.time() - t))

    # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Skeleton', frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    print("total = {}".format(time.time() - t))

    vid_writer.write(frame)

vid_writer.release()