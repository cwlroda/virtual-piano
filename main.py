import cv2
from piano import Piano
from threading import Thread

piano = Piano()

def playMusic(event, x, y, flags, params):
  if event == cv2.EVENT_MOUSEMOVE:
    Thread(target=piano.press, args=(x,y,)).start()
    # key = piano.press(x, y)
    # if key is not None:
    #   print(key+1)


def main():
  vid = cv2.VideoCapture(0)

  while True:
    cv2.namedWindow("Piano")
    ret, frame = vid.read()
    piano.display(frame)
    frame = cv2.flip(frame, 1)
    cv2.setMouseCallback("Piano", playMusic)
    cv2.imshow("Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  vid.release()
  cv2.destroyAllWindows()

main()