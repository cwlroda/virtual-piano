import cv2
import numpy as np

from tkinter import *

class Piano:
  def __init__(self, scale=[0.8, 0.2], octaves=2):
    self.scale = scale
    self.width = scale[0]
    self.height = scale[1]
    self.octaves = octaves
    self.keys = self.octaves*8
    self.whiteKeys = {}
    self.blackKeys = {}

  def display(self, img):
    self.drawRectangle(img)
    self.drawLines(img)
    self.keyboard = img[self.up:self.down, self.left:self.right]

  def drawRectangle(self, img):
    height, width, _ = img.shape
    self.left = int(width*(1-self.width)/2)
    self.right = int(width*(1+self.width)/2)
    self.up = int(height*(1-self.height))
    self.down = int(height)
    self.fadedBox(img, self.up, self.down, self.left, self.right, white=True)

  def drawLines(self, img):
    cv2.rectangle(img, (self.left, self.up), (self.right, self.down), (0,0,0), thickness=5)
    length = self.right-self.left
    x_vals = [int(length*i/self.keys) for i in range(1, self.keys)]
    for index, i in enumerate(x_vals):
      cv2.line(img, (self.left+i, self.up), (self.left+i, self.down), (0,0,0), 1)

    keys = '0111011'
    halfKeyWidth = int(0.25*length/self.keys)
    self.blackKeyWidth = halfKeyWidth*2
    self.blackKeyHeight = int(0.5*(self.up-self.down))
    for index, i in enumerate(x_vals):
      if int(keys[index%len(keys)]):
        self.fadedBox(img, self.up, self.up-self.blackKeyHeight, self.left+i-halfKeyWidth, self.left+i+halfKeyWidth, white=False)
  
  def fadedBox(self, img, up, down, left, right, white=True):
    sub_img = img[up:down, left:right]
    
    color = 255 if white else 0
    rect = np.ones(sub_img.shape, dtype=np.uint8) * color
    res = cv2.addWeighted(sub_img, 0.5, rect, 0.5, 1.0)

    img[up:down, left:right] = res
      


