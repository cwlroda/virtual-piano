import cv2
import numpy as np
from playsound import playsound
import os

from tkinter import *

class Piano:
  def __init__(self, scale=[0.9, 0.25], octaves=2):
    self.scale = scale
    self.width = scale[0]
    self.height = scale[1]
    self.octaves = octaves
    self.keys = self.octaves*8 -1
    self.whiteKeys = []
    self.pressed = False
    self.playing = None

  def display(self, img):
    self.img = img
    height, width, _ = self.img.shape
    self.left = int(width*(1-self.width)/2)
    self.right = int(width*(1+self.width)/2)
    self.up = int(height*(1-self.height))
    self.down = int(height)
    self.drawRectangle()
    self.drawLines()
    if self.pressed:
      self.solidBox(self.playing[1][0], self.playing[1][1], self.playing[0][0], self.playing[0][1])

  def drawRectangle(self):
    self.fadedBox(self.up, self.down, self.left, self.right, white=True)

  def drawLines(self):
    cv2.rectangle(self.img, (self.left, self.up), (self.right, self.down), (0,0,0), thickness=5)
    length = self.right-self.left
    x_vals = [int(length*i/self.keys) for i in range(0, self.keys+1)]

    for i in range(len(x_vals)-1):
      if len(self.whiteKeys) < self.keys:
        self.whiteKeys.append(((self.left+x_vals[i], self.left+x_vals[i+1]), (self.up, self.down)))
      if i != 0:
        cv2.line(self.img, (self.left+x_vals[i], self.up), (self.left+x_vals[i], self.down), (0,0,0), 1)

    # keys = '0111011'
    # halfKeyWidth = int(0.25*length/self.keys)
    # self.blackKeyWidth = halfKeyWidth*2
    # self.blackKeyHeight = int(0.5*(self.up-self.down))
    # for index, i in enumerate(x_vals):
    #   if 0 < index < self.keys and int(keys[(index-1)%len(keys)]):
    #     self.fadedBox(self.up, self.up-self.blackKeyHeight, self.left+i-halfKeyWidth, self.left+i+halfKeyWidth, white=False)
  
  def fadedBox(self, up, down, left, right, white=True):
    sub_img = self.img[up:down, left:right]

    color = 255 if white else 0
    rect = np.ones(sub_img.shape, dtype=np.uint8) * color
    res = cv2.addWeighted(sub_img, 0.5, rect, 0.5, 1.0)

    self.img[up:down, left:right] = res

  def solidBox(self, up, down, left, right):
    sub_img = self.img[up:down, left:right]

    rect = np.zeros(sub_img.shape, dtype=np.uint8)
    rect[:,:,2] = 255
    res = cv2.addWeighted(sub_img, 0.5, rect, 0.5, 1.0)

    self.img[up:down, left:right] = res

  def press(self, x, y):
    for position, key in enumerate(self.whiteKeys):
      if between(x, y, key):
        self.pressed = True
        # for actual fingers
        # # self.playing = key
        
        # for mouse
        self.playing = self.whiteKeys[self.keys-position-1]
        filename = "data\keys" + "\\" + str(position+1) + ".mp3"
        filedir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(filedir, filename)
        playsound(filepath)
        return position
    else:
      self.pressed = False
      self.playing = None
      return None
      

def between(x, y, box):
  return box[0][0] < x < box[0][1] and box[1][0] < y < box[1][1]
      


