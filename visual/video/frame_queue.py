import time
import subprocess
import random

from os import listdir
from os.path import isfile, join as join_paths
from random import shuffle
from itertools import count
from collections import deque

import imutils

import cv2 as cv
import numpy as np

from utils.errors import MissingImageCategoryError
from utils.constants import DEFAULT_FPS


class FrameQueue:
  paths = []
  current_file_index = 0
  current_image_index = 0
  current_image_frame = 0
  black_frame = None
  loaded_image = None
  loaded_video = None
  loaded_videos = []
  loaded_images = []
  loaded_category_images = dict()
  loaded_category_videos = dict()
  master_height = 1000
  master_width = 1000
  master_size = (master_width, master_height)
  video_source_rate = DEFAULT_FPS*8
  image_source_rate = 3
  source_rate = 30
  video_enabled = False
  current_category = None
  main_categories = []
  peak_categories = []
  current_categories = []
  current_peak_categories = []
  frame_count = 0
  fps = DEFAULT_FPS
  backgrounds = dict()
  interpolation = 'source_rate'
  image_enabled = True

  def __init__(self, file_paths, height=None, width=None, image_source_rate=None, video_source_rate=None, fps=None):
    # master_height = 720
    # master_width = 1280
    # master_height = 1080
    # master_width = 1920
    self.fps = fps if fps else DEFAULT_FPS
    self.frame_count = 0
    self.paths += file_paths
    self.get_file_paths()
    self.black_frame = np.zeros((self.master_height,self.master_width,3), np.uint8)
    if video_source_rate:
      self.video_source_rate = video_source_rate
    if image_source_rate:
      self.image_source_rate = image_source_rate
    if height and width:
      self.master_height = height
      self.master_width = width
      self.master_size = (self.master_width, self.master_height)
    self.video_enabled = True
    self.image_enabled = False

  def destroy(self):
    if self.loaded_video:
      self.loaded_video.release()


  def get_file_paths(self):
    # >>> import glob
    # >>> glob.glob('./[0-9].*')
    # ['./1.gif', './2.txt']
    # >>> glob.glob('*.gif')
    # ['1.gif', 'card.gif']
    # >>> glob.glob('?.gif')
    # ['1.gif']
    for p in self.paths:  
      only_files = [join_paths(p, f) for f in listdir(p) if isfile(join_paths(p, f))]
      only_videos = list(filter(lambda f: '.mp4' in f.lower() or '.mov' in f.lower(), only_files))
      only_images = list(filter(lambda f: '.jpg' in f.lower() or '.jpeg' in f.lower(), only_files))
      self.loaded_images += only_images
      self.loaded_videos += only_videos
      self.all_files = self.loaded_images + self.loaded_videos
      shuffle(self.all_files)


  def load_image_categories(self, categories):
    for c in categories:
      if c not in self.loaded_category_images.keys():
        self.loaded_category_images[c] = [ p for p in self.loaded_images if all([w in p for w in c.split(' ')]) ]
    for c in categories:
      if len(self.loaded_category_images[c]) == 0:
        raise MissingImageCategoryError(c+' has no images')

  
  def load_video_categories(self, categories):
    for c in categories:
      if c not in self.loaded_category_videos.keys():
        self.loaded_category_videos[c] = [ p for p in self.loaded_videos if all([w in p for w in c.split(' ')]) ]
    for c in categories:
      if len(self.loaded_category_videos[c]) == 0:
        raise MissingImageCategoryError(c+' has no videos')


  def get_image_frame(self, category=None):
    # self.current_images = [1,2,3]
    image_file_list = self.loaded_category_images[category] if category else self.loaded_images

    if self.current_image_frame%self.image_source_rate == 0:
      self.next_image_index = random.randint(0,len(image_file_list)-1)
      if self.next_image_index == self.current_image_index:
        self.next_image_index += 1
        if self.next_image_index == len(image_file_list):
          self.next_image_index = 0
      image_path = image_file_list[self.next_image_index]
      self.loaded_image = cv.imread(image_path)
      self.current_file_index = self.next_image_index
    self.current_image_frame += 1
    return self.loaded_image

  def reverse_queue_frame(self):
    reversed_frames = 0 # will not work
    if self.frame_count%reversed_frames == 0:
      last_frames = []
    if self.frame_count%reversed_frames < (reversed_frames/2):
      last_frames.append(frame)
    if self.frame_count%reversed_frames >= (reversed_frames/2):
      reversed_index =(self.frame_count%reversed_frames) - (len(last_frames)-1)
      frame = last_frames[-reversed_index]  

  def enable_video_check(self):
    # if frame_count%(self.fps * 6) == 0:
    #   if random.randint(0, 10) > 4:
    #     self.video_enabled = True

    if self.frame_count%8 == 0:
      self.video_enabled = not self.video_enabled
    # if frame_count%self.fps == 0:
    #   if random.randint(0, 10) > 5:
    #     self.video_enabled = False

  def prepare_frame(self, count, video=True):
    self.frame_count = count
    if video and not self.loaded_video:
      self.load_next_video(category=self.current_category)
    if self.interpolation == 'source_rate' and self.frame_count%self.source_rate==0:
      self.image_enabled = not self.image_enabled
      self.video_enabled = not self.video_enabled
    if video:
      if self.video_enabled and self.frame_count%self.video_source_rate == 0:
        self.load_next_video(category=self.current_category)


  def update_source_rate(self, source_rate, image_source_rate=None, interpolation=None):
    self.video_source_rate = source_rate
    self.image_source_rate = source_rate
    if image_source_rate:
      self.image_source_rate = image_source_rate
    if interpolation:
      self.interpolation = 'source_rate'


  def get_image(self, category):
    frame = self.get_image_frame(category=category)
    none_frame_count = 0
    while frame is None:
      none_frame_count += 1
      frame = self.get_image_frame(category=category)
      if none_frame_count > DEFAULT_FPS:
        print('IMAGES NOT WORKING:')
        none_frame_count = 0
    return frame, 0

  def get_video_frame(self, category):
    ret, frame = self.loaded_video.read()
    none_frame_count = 0
    while frame is None:
      none_frame_count += 1
      ret, frame = self.loaded_video.read()
      if none_frame_count > DEFAULT_FPS:
        print('video not working:', self.loaded_videos[self.current_file_index])
        self.load_next_video()
        none_frame_count = 0
    return frame, self.loaded_video.get(cv.CAP_PROP_POS_FRAMES)

  def get_background_frame(self, color):
    frame = self.black_frame
    # background True=black or color: (0,255,0)
    if len(color) == 3:
      frame
      if color not in self.backgrounds.keys():
        self.backgrounds[color] = frame
        self.backgrounds[color][:,:] = color
      frame = self.backgrounds[color]
    # blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
    # multi frame(or color) in frame 
    # blank_image[:,width//2:width] = (0,255,0)
    return frame, 0

  def get_frame(self, image=False, background=False, category=None, both=False):
    if background:
      return self.get_background_frame(background)
    elif image:
      return self.get_image(category)
    elif both and self.image_enabled:
      return self.get_image(category)
    return self.get_video_frame(category)
    


  def load_next_working_video(self):
    while True:
      if frame.shape[0] <= self.master_height and frame.shape[1] <= self.master_width:
        break

  def load_next_video(self, category=None):
    if category:
      video_file_list = self.loaded_category_videos[category]
    else:
      video_file_list = self.loaded_videos

    if self.loaded_video is not None:
      self.loaded_video.release()
      self.loaded_video = None

    self.next_index = random.randint(0,len(video_file_list)-1)
    if self.next_index == self.current_file_index:
      self.next_index += 1
      if self.next_index == len(video_file_list):
        self.next_index = 0
  
    path = video_file_list[self.next_index]
    # if '.mov' in file_path.lower() or '.mp4' in file_path.lower():
      # self.print_video_segment(file_path, 2)
      # valid_file = True
    self.loaded_video = cv.VideoCapture(path)
    self.current_file_index = self.next_index
  

  # def print_video_segment(self, path, seconds=0, duration_frames=300):
  #   frame_count = 0
  #   current_seconds = 0
  #   none_frame_count = 0
  #   dd = cv.VideoCapture(path)
  #   x, y = 200, 200
  #   while True:
  #     ret, frame = dd.read()
  #     if frame is None:
  #         none_frame_count += 1
  #         if none_frame_count > 100:
  #           break
  #         continue
  #     else:
  #       frame_count += 1
  #       frame = frame[y:y+self.master_height, x:x+self.master_width]
  #       if frame_count%30 == 0:
  #         self.change_current_video_filter()
  #       final_frame = self.apply_video_filters(frame)
  #       if seconds > 0:
  #         current_seconds = frame_count/self.fps
  #         if current_seconds >= seconds:
  #           break
  #       elif frame_count > duration_frames:
  #           break
  #       self.video_writer.write(final_frame) 
  #       self.total_frames += 1
        
  #   # print(none_frame_count)
  #   dd.release()

  # def print_image_segment(self, path, seconds=0, duration_frames=100):
  #   frame_count = 0
  #   current_seconds = 0
  #   none_frame_count = 0
  #   image = cv.imread(path)
  #   image_height = image.shape[0]
  #   image_width = image.shape[1]
  #   x, y = 100, 200
  #   while frame_count < duration_frames:
  #     frame_count += 1
  #     self.total_frames += 1

  #     cut_y = y+self.master_height
  #     if cut_y+frame_count <= image_height: 
  #       y += frame_count
  #       cut_y += frame_count
  #     cut_x = x+self.master_width
  #     if cut_x+frame_count <= image_width: 
  #       x += frame_count
  #       cut_x += frame_count
  #     # else:

  #     frame = image[y:cut_y, x:cut_x]
  #     if frame_count%30 == 0:
  #       self.change_current_image_filter()
  #     final_frame = self.apply_image_filters(frame)
  #     # print('image',final_frame)
  #     self.video_writer.write(final_frame) 
  #     if none_frame_count > 50:
  #       break
  #     if seconds > 0:
  #       current_seconds = frame_count/self.fps
  #       if current_seconds >= seconds:
  #         break


