import cv2
import time
import numpy as np
from piano import Piano
from threading import Thread
from UMatFileVideoStream import UMatFileVideoStream

piano = Piano()

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

threshold = 0.2

input_source = "input.mp4"
# cap = cv2.VideoCapture(0)
# hasFrame, frame = cap.read()

def playMusic(x, y):
    Thread(target=piano.press, args=(x,y,)).start()

if __name__ == '__main__':
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
    points = np.zeros(5, dtype=(int,2))
    old_points = np.zeros(5, dtype=(int,2))
    frame_counter = 0
    frame_gap = 10

    while 1:
        t = time.time()

        # hasFrame, frame = cap.read()

        frame = cap.read()
        # frameCopy = np.copy(frame)
        frameCopy = frame
        
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
        print("Time taken for net = {}".format(time.time() - t))

        piano.display(frame)
        
        # Empty list to store the detected keypoints
        fingertips = [4, 8, 12, 16, 20]
        num = None

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

                if int(point[1]) - old_points[finger][1] > 20:
                    playMusic(int(point[0]), int(point[1]))
                    num = finger + 1
                    print("Finger {} pressed".format(num))
                
            # print('frame number:', frame_counter)
            # print('finger number:', finger)
            # print('Points:', points)
        cv2.putText(frame, "Finger {} pressed".format(num), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # Draw Skeleton
        # for pair in POSE_PAIRS:
        #     partA = pair[0]
        #     partB = pair[1]

        #     if points[partA] and points[partB]:
        #         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
        #         cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #         cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # print("Time Taken for frame = {}".format(time.time() - t))

        # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)
        # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        frame_counter += 1
        time_taken = time.time() - t
        print("total = {}".format(time_taken))
        fps = 1.0 / time_taken
        print("fps= {}".format(fps))
        
        vid_writer.write(frame)

    vid_writer.release()