import cv2
from piano import Piano

piano = Piano()

def playMusic(event, x, y, flags, params):
  if event == cv2.EVENT_MOUSEMOVE:
    key = piano.press(x, y)
    if key is not None:
      print(key+1)

def main():
  input_source = 'input.mp4'
  vid = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)

  while True:
    cv2.namedWindow("Piano")
    ret, frame = vid.read()

    try:
      piano.display(frame)
      frame = cv2.flip(frame, 1)
      cv2.setMouseCallback("Piano", playMusic)
      cv2.imshow("Piano", frame)
    except:
      print('end of video')
      break

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
    cv2.waitKey(30)

  vid.release()
  cv2.destroyAllWindows()

main()