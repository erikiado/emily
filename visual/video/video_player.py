import cv2 as cv
import numpy as np

from os import listdir
from os.path import isfile, join as join_paths
# from bisect import bisect_left # binary/dichotomic search on lists

import time
import subprocess
import random
import pygame
import ffmpeg
import sys
from random import shuffle
from itertools import count

from collections import deque
from imutils.video import VideoStream
# import argparse
import imutils

from PIL import ImageFont, ImageDraw, Image, ImageOps


# cv.VideoWriter(filename, fourcc, fps, (w, h), ...)
# frame = np.zeros((h, w), ...)



sys.path.append('/Users/erikiado/Code/emily/audio')
from audio import AudioAnalyzer


print(cv.__version__)

#  TODO: filters 
# https://docs.opencv.org/4.2.0/d4/d86/group__imgproc__filter.html
# https://docs.opencv.org/4.2.0/db/d28/tutorial_cascade_classifier.html
#  TODO: create instagram post image and video


#audio analyzer class
# deteccion de picos en cancion
# get current volume
# duracion de cancion
# conexion con interfaz



class Palette():
    def __init__(self, palette_img_src, brightness_func):
        self.palette = {}
        with Image.open(palette_img_src) as palette_img:
            palette_img_height = palette_img.size[1]
            list(tqdm(self._build_palette(palette_img, brightness_func), total=palette_img_height))
        self.sorted_keys = sorted(self.palette.keys())
    def _build_palette(self, palette_img, brightness_func):
        width, height = palette_img.size
        palette_pixels = palette_img.load()  # getting PixelAccess
        for j in range(height):
            for i in range(width):
                brightness = brightness_func(palette_pixels[i, j])
                self.palette[brightness] = palette_pixels[i, j] # nothing smart (ex: avg): we only keep the last brightness value processed
            yield 'ROW_COMPLETE' # progress tracking
    def __getitem__(self, key):
        i = bisect_left(self.sorted_keys, key)  # O(logN)
        return self.palette[self.sorted_keys[i]]



class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class MissingImageCategoryError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.expression = 'Missing category images'
        self.message = message


# dimensiones dependendientes del source
#  lo mas grande con posibilidad de movimiento
# duracion en tiempo; ongoing? no guardar?
# ajustar el video a hd
# detectar bordes al color


# opengl
# https://rdmilligan.wordpress.com/2015/08/29/opencv-and-opengl-using-python/


# orden de videos, seleccionar n videos y reutilizarlos siguiendo alguna estructura

# pool de efectos? basados en estructura de secuencia de videos, tiempo, sonido

# efectos con mascaras, thresholds y colores


class RandomText:
  stroke = None
  size = None
  x = None
  y = None
  r = None
  g = None
  b = None
  text = None
  master_height = 0
  master_width = 0
  test_run = False
  words = ['erikiado', 'erikiano', '@@@@@@', '######', '%%%%%%%', 'sube','baja','detente','sigue','escuchame','callate','sigueme','vete','explora']
  output_name = 'output.mp4'


  def __init__(self, master_width, master_height, test_run=False):
    self.master_height = master_height
    self.master_width = master_width
    self.update()
    self.text = ''
    self.test_run = test_run
  
  def update(self, text=None, stroke=None, size=None):
    self.x = random.randint(0,self.master_width - int(self.master_width/4))
    self.y = random.randint(0,self.master_height - int(self.master_height/5))
    self.r = random.randint(0,255)
    self.g = random.randint(0,255)
    self.b = random.randint(0,255)
    if size:
      self.size = size
    else:
      self.size = random.randint(1,4)
    if stroke:
      self.stroke = stroke
    else:
      self.stroke = random.randint(1,5)
    if text:
      self.text = text
    else:
      next_index = random.randint(0,len(self.words)-1)
      self.text = self.words[next_index]
  
  def put(self,frame):
    cv.putText(frame, self.text, (self.x, self.y), cv.FONT_HERSHEY_SIMPLEX, self.size , (self.r,self.g,self.b), self.stroke, cv.LINE_AA)



class VideoEditor:
  video_writer = None
  clock = None
  paths = []
  current_text = None
  current_lyric = None
  lyrics_enabled = False
  current_file_index = 0
  current_image_index = 0
  current_image_frame = 0
  loaded_image = None
  loaded_video = None
  lyrics = []
  loaded_fonts = []
  loaded_wavs = []
  loaded_videos = []
  loaded_images = []
  fonts = []
  loaded_category_images = dict()
  loaded_category_videos = dict()
  color_video = True
  total_frames = 0
  master_height = 1000
  master_width = 1000
  current_lyric_color = None
  # master_height = 720
  # master_width = 1280
  filters_enabled = True
  # master_height = 1080
  # master_width = 1920
  master_size = (master_width, master_height)
  fps = 30
  video_source_rate = fps*8
  image_source_rate = 3
  lyric_text_rate = 4
  peak_lyric_text_rate = 10
  global_peaks_enabled = True
  video_enabled = False
  colors = []
  font_colors = []
  peak_duration = 6

  fourcc = None
  backSub = None
  video_filters = ['none','invert','backsub','backsub-color','backsub-color-invert','backsub-color-color']
  image_filters = ['none','invert','change-colors']
  current_video_filter = 'none'
  current_image_filter = 'none'

  current_category = None
  main_categories = []
  peak_categories = []
  current_categories = []
  current_peak_categories = []
  fonts_path = None

  def __init__(self, file_paths, fonts_path=None,height=None, width=None, image_source_rate=None, video_source_rate=None, peak_duration=None, fps=None, test_run=False):
    self.paths += file_paths
    self.get_file_paths()
    self.video_writer = None
    # self.video_writer = cv.VideoWriter()
    self.fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # self.fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    # self.fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    self.backSub = cv.createBackgroundSubtractorMOG2()
    self.clock = pygame.time.Clock()
    self.current_text = 'erikiano'
    self.filters_enabled = True
    self.test_run = test_run
    if fps:
      self.fps = fps
    if fonts_path:
      self.fonts_path = fonts_path
      self.load_fonts()
    if video_source_rate:
      self.video_source_rate = video_source_rate
    if image_source_rate:
      self.image_source_rate = image_source_rate
    if peak_duration:
      self.peak_duration = peak_duration
    if self.test_run:
      print('test run')
      print('test run')
      print('test run')
      self.master_height = 300
      self.master_width = 300
      self.master_size = (self.master_width, self.master_height)
    if height and width:
      self.master_height = height
      self.master_width = width
      self.master_size = (self.master_width, self.master_height)

      # self.filters_enabled = False


    # self.backSub = cv.createBackgroundSubtractorKNN()

  def load_fonts(self, size=50, variation=0.1):
    self.loaded_fonts = [join_paths(self.fonts_path, f) for f in listdir(self.fonts_path) if isfile(join_paths(self.fonts_path, f)) and 'tf' in f ]
    print('font files:',self.loaded_fonts)
    for p in self.loaded_fonts:
      font_name = p.split('/')[-1].split('.')[0]
      font_widths = {
        'Hello Chilly': 2,
        'JMH Typewriter-Bold': 1,
        'JMH Typewriter': 1,
        'THRASHER PERSONAL USE': 1,
        'Distort Me': 2,
        'Distortion Dos Analogue': 2,
        'A-cuchillada-font-ffp': 2,
        'CFAnarchy-Regular': 2,
        '28 Days Later': 2,
        '1942': 2,
        'Nervous': 2,
      }
      font_widths = {
        'Finger Printed': 1,
        'BLACK AREA': 1,
        'Harukaze': 1,
        'ChineseDragon': 1,
        'Kashima Demo': 1,
        'Ghost Factory': 1,
        'Bad Signal': 1,
        'CFGlitchCity-Regular': 1,
        'Doctor Glitch': 1,
        'doctor punk': 1,
        'Romanica': 1,
        'Fuji Quake Zone': 1,
        'Championship': 1,
        'The Drunked Man St': 1,
        'I believe in life before death': 1,
        'Snake Jacket': 1,
        'HighVoltage Rough': 1,
        'messagefromtheeast': 1,
        'War is Over': 1,
        'ExtraBlur': 1,
        'Experimento': 1,
        'Onnenmyyra': 1,
        'KGBytheGraceofGod': 1,
        'Static-2': 1,
        'kandinsky': 1,
        'Bad Coma': 1,
        'MetalBlockUltra': 1,
        'Manga Style': 1,
        'helvetica-destru-pix': 1,
        'CFNelsonOldNewsPaper': 1,
        'Rythm N Blacks': 1,
        'electroharmonix': 1,
        'black spiral': 1,
        'NEWYORK': 1,
        'Gelio Retsina': 1,
        'Gelio Pasteli': 1,
        'Gelio Kleftiko': 1,
        'Gelio Greek Diner': 1,
        'Gelio Fasolada': 1,
        'ProblematicPiercer': 1,
        'Problematic Piercer': 1,
        'memories': 1,
        'Bandung Hardcore GP': 1,
        'EASTRIAL': 1,
        'EASTRIAL': 1,
        'Ming Imperial': 1,
        'Vtks Relaxing Blaze': 1,
        'youmurdererbb_reg': 1,
        'Pulse_virgin': 1,
        'go3v2': 1,
        'PoseiAOE': 1,
        'brushed': 1,
        'leadcoat': 1,
        'Plaq - 108': 1,
        'ANUNE___': 1,
        'ANUNEDW_': 1,
        'Fluox___': 1,
        'DIOGENES': 1,
        'WISHFULWAVES': 1,
        '4990810_': 1,
        'EMPORO': 1,
        'glashou': 1,
        'ASS': 1,
        'Acidic': 1,
      }
      big = size-int((size*variation)-(font_widths[font_name]*(size//10)))
      small = size+int(size*variation)
      if small > big:
        tmp_small = big
        big = small
        small = tmp_small
      font_size = random.randint(small, big)
      font = ImageFont.truetype(p, font_size)
      font_width = font_widths[font_name] * (0.8 * font_size)
      self.fonts.append(dict(font=font,width=font_width))

  def destroy_video_writer(self):
    self.video_writer.release()

  def get_file_paths(self):
    for p in self.paths:  
      only_files = [join_paths(p, f) for f in listdir(p) if isfile(join_paths(p, f))]
      only_videos = list(filter(lambda f: '.mp4' in f.lower() or '.mov' in f.lower(), only_files))
      only_images = list(filter(lambda f: '.jpg' in f.lower() or '.jpeg' in f.lower(), only_files))
      only_wavs = list(filter(lambda f: '.wav' in f.lower(), only_files))
      self.loaded_wavs += only_wavs
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
    image_file_list = self.loaded_category_images[category]    

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

  def get_frame(self, image=False, category=None):
    if image:
      frame = self.get_image_frame(category=category)
      return frame, 0
    ret, frame = self.loaded_video.read()
    return frame, self.loaded_video.get(cv.CAP_PROP_POS_FRAMES)

  def tearY(self, frame):
    size = frame.shape
    sizex = size[1]
    block = sizex/10
    blockx = random.random()*10*block
    frame[:,blockx:blockx+block,:] = 0
    return frame

  def timestamp_to_frame(self, ts):
    minute, s_ms = ts.split(':')
    seconds, ms = s_ms.split('.')
    minute, seconds, ms = int(minute), int(seconds), int(ms)
    time_frame = ((minute * 60) + seconds) * self.fps
    time_frame += round(ms*self.fps/1000)
    return time_frame

      
  def edit_music_image_gallery(self, audio_file_path, main_categories, peak_categories=[], lyrics=None, episodes=[]):
    if lyrics:
      self.lyrics = []
      print(lyrics)
      with open(lyrics, 'r') as f:
        for l in f.readlines():
          if '(' in l  and ')' in l and '-' in l:
            lyric_line = dict(start=self.timestamp_to_frame(l[1:10]),end=self.timestamp_to_frame(l[11:20]),text=l[22:].rstrip())
            self.lyrics.append(lyric_line)
    anal = AudioAnalyzer()
    print('loading:', audio_file_path)
    self.output_name = audio_file_path.split('/')[-1].split('.')[0] + '.mp4'
    audio_duration, peak_frames = anal.get_peaks_for_audiovisual(audio_file_path, self.fps)
    print('audio duration:',str(audio_duration),'seconds')
    print('audio duration:',str(audio_duration/60),'minutes')
    total_video_frames = int(audio_duration*self.fps)
    print('audio duration:',str(total_video_frames),'frames')
    print('peaks/minutes:',str(int(len(peak_frames)/(audio_duration/60))),'bpm?')

    frame_count = 0
    none_frame_count = 0
    x, y = 100, 200
    # freetype = cv.createFreeType2()
    # random_text = RandomText(self.master_width,self.master_height,test_run=self.test_run)
    reversed_frames = 5 # N frames forward then reversed, 5 frames skipped
    reversed_frames = reversed_frames*2
    
    peak_frame_count = 0
    peaks_enabled = False

    print('working on:', self.output_name)
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size, 
                                       self.color_video)
    self.main_categories = main_categories
    self.peak_categories = peak_categories
    self.current_categories = self.main_categories.copy()
    self.current_peak_categories = self.peak_categories.copy()

    episodes_categories = []
    for e in episodes:
      if 'categories' in e['options']:
        episodes_categories += e['options']['categories']
      if 'peak_categories' in e['options']:
        episodes_categories += e['options']['peak_categories']
    all_categories = list(set(main_categories+peak_categories+episodes_categories))
    print('all categories: ',all_categories)
    self.load_image_categories(all_categories)
    self.load_video_categories(all_categories)
    self.current_category = self.current_categories[0]
    self.peak_duration = 6
    self.image_source_rate = 6
    episode_starts = [ e['start'](total_video_frames) if 'function' in str(type(e['start'])) else e['start'] for e in episodes ]
    lyric_starts = [ l['start'] for l in self.lyrics ]
    lyric_ends = [ l['end'] for l in self.lyrics ]
    cv_im_rgb = np.zeros((self.master_height,self.master_width,3), np.uint8)
    pil_im = Image.fromarray(cv_im_rgb)  
    draw = ImageDraw.Draw(pil_im)  
    # for l in self.lyrics:
    #     print(l['text'])
    #     for p in self.loaded_fonts:
    #       font = ImageFont.truetype(p, 40)
    #       print(font)
    #       print(str(font))
    #       w, h = draw.textsize(l['text'])
    #       # print(p)
    #       # print(self.master_width>w,w,h)
    # return
    print('episodes:', episode_starts)
    print('lyrics:', lyric_starts)

    self.colors = [
      (0,0,0),
      (255,255,255),
      (0,0,255),
    ]
    self.font_colors = [
      (0,0,0),
      (255,255,255),
      (255,255,255),
      (255,255,255),
      (255,255,255),
      (255,255,255),
      (255,0,0),
      (255,0,0),
      (255,0,0),
      (255,0,0),
    ]
    
    while frame_count != total_video_frames:
      # self.clock.tick(cycles_per_second)
      if frame_count%self.video_source_rate == 0:
        self.load_next_video(category=self.current_category)
      if frame_count in episode_starts:
        for i, e in enumerate(episodes):
          if episode_starts[i] == frame_count:
            options = e['options']
            if 'peak_duration' in options:
              self.peak_duration = options['peak_duration']
            if 'image_source_rate' in options:
              self.image_source_rate = options['image_source_rate']
            if 'lyric_text_rate' in options:
              self.lyric_text_rate = options['lyric_text_rate']
            if 'peaks_enabled' in options:
              self.global_peaks_enabled = options['peaks_enabled']
            else:
              self.global_peaks_enabled = True
            if 'categories' in options:
              self.current_categories = self.main_categories.copy()
              self.current_categories += options['categories']
            if 'peak_categories' in options:
              self.current_peak_categories = self.peak_categories.copy()
              self.current_peak_categories += options['peak_categories']


      if frame_count in peak_frames:
        peaks_enabled = True
        peak_frame_count = 0
        self.current_category = random.choice(self.current_peak_categories)
      if peak_frame_count < self.peak_duration and peaks_enabled:
        peak_frame_count += 1 
      if peak_frame_count == self.peak_duration:
        peaks_enabled = False
        self.current_category = random.choice(self.current_categories)
        peak_frame_count = 0
        if random.randint(0,10) > 6:
          self.change_current_video_filter('none')
      
      if frame_count in lyric_ends:
        for i, l in enumerate(self.lyrics):
          if lyric_ends[i] == frame_count:
            self.lyrics_enabled = False
      
      if frame_count in lyric_starts:
        for i, l in enumerate(self.lyrics):
          if lyric_starts[i] == frame_count:
            self.lyrics_enabled = True
            self.current_lyric = l['text']

      # if frame_count%(self.fps * 6) == 0:
      #   if random.randint(0, 10) > 4:
      #     self.video_enabled = True

      if frame_count%8==0:
        self.video_enabled = not self.video_enabled

      # if frame_count%self.fps == 0:
      #   if random.randint(0, 10) > 5:
      #     self.video_enabled = False

      frame, current_frame = self.get_frame(image=(not self.video_enabled), category=self.current_category)
      # if frame_count%reversed_frames == 0:
      #   last_frames = []
      # if frame_count%reversed_frames < (reversed_frames/2):
      #   last_frames.append(frame)
      # if frame_count%reversed_frames >= (reversed_frames/2):
      #   reversed_index =(frame_count%reversed_frames) - (len(last_frames)-1)
      #   frame = last_frames[-reversed_index]      
      
      while frame is None:
        none_frame_count += 1
        frame, current_frame = self.get_frame(image=(not self.video_enabled), category=self.current_category)
        if none_frame_count > 60 and self.video_enabled:
          print('video not working:', self.loaded_videos[self.current_file_index])
          self.load_next_video()
          none_frame_count = 0
      
      # if frame_count%self.video_source_rate == 0:
      #   print(self.loaded_videos[self.current_file_index])
      #   print(frame.shape)

      # shape = frame.shape
      # int(shape[0]/2) - int(self.master_height/2)
      # int(shape[0]/2) + int(self.master_height/2)
      # int(shape[1]/2)
      # frame = frame[int(shape[0]/2) - int(self.master_height/2):int(shape[0]/2) + int(self.master_height/2), int(shape[1]/2) - int(self.master_width/2):int(shape[1]/2) + int(self.master_width/2)]
      # frame = frame[y:y+self.master_height, x:x+self.master_width]

      if random.randint(0,10) > 8:
        if frame_count%int(self.image_source_rate/2) == 0 or frame_count in peak_frames:
          self.change_current_image_filter()
        if frame_count%int(self.video_source_rate/2) == 0 or frame_count in peak_frames:
          self.change_current_video_filter()

      if frame_count%15 == 0:
        self.change_current_text()
      # if frame_count%10 == 0:
      #   rand_text_size = random.randint(1,2)
      #   if self.master_height < 350:
      #     rand_text_size = 0.5
      #   # if test_run:
      #     # random.randint(1,2) #float
      #   random_text.update(self.current_text,stroke=1,size=random.randint(1,2))

      # final_frame = frame
      # final_frame = self.apply_video_filters(frame)

      # scale_percent = 220 # percent of original size
      # width = int(img.shape[1] * scale_percent / 100)
      # height = int(img.shape[0] * scale_percent / 100)
      # dim = (width, height)

      # $targetWidth = $targetHeight = min($size, max($originalWidth, $originalHeight));

      # if ($ratio < 1) {
      #     $targetWidth = $targetHeight * $ratio;
      # } else {
      #     $targetHeight = $targetWidth / $ratio;
      # }

      # $srcWidth = $originalWidth;
      # $srcHeight = $originalHeight;
      # $srcX = $srcY = 0;

      # # resize image
      # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
      shape = frame.shape
      if shape[1] > shape[0]:
        # encontrar ratio de height
        ratio = self.master_height/shape[0]
      else:
        # encontrar ratio de width
        ratio = self.master_width/shape[1]
      frame = cv.resize(frame, (int(shape[1]*ratio),int(shape[0]*ratio)))#, fx=ratio, fy=ratio)#(int(self.master_width* ratio) , int(self.master_height*ratio)))
      shape = frame.shape
      # to fit?
      frame = frame[int(shape[0]/2) - int(self.master_height/2):int(shape[0]/2) + int(self.master_height/2), int(shape[1]/2) - int(self.master_width/2):int(shape[1]/2) + int(self.master_width/2)]
      final_frame = cv.resize(frame, (self.master_width, self.master_height))
      # final_frame = frame
      final_frame = self.apply_image_filters(final_frame)

      # cv.putText(final_frame, str(frame_count)+self.current_text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (256,256,256))
      # if frame_count % 10 > 5:
        # cv.putText(final_frame, self.current_text, (200, 300), cv.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255))
      if self.lyrics_enabled:
        cv_im_rgb = cv.cvtColor(final_frame,cv.COLOR_BGR2RGB)  
        pil_im = Image.fromarray(cv_im_rgb)  
        draw = ImageDraw.Draw(pil_im)  
        if frame_count%self.lyric_text_rate==0 or not self.current_lyric_color or (self.global_peaks_enabled and peaks_enabled and frame_count%self.peak_lyric_text_rate==0):
          # previous_lyric_frame = 
          self.current_lyric_color = random.choice(self.font_colors)
          # lyric_text_size = random.randint(1)
          lyric_text_size = random.uniform(.5, 1.2)
          lyric_text_stroke = random.randint(1,2)
          # cv.putText(final_frame, self.current_lyric, (lyric_text_x, lyric_text_y), cv.FONT_HERSHEY_SIMPLEX, lyric_text_size, self.current_lyric_color, lyric_text_stroke, cv.LINE_AA)
          ###############################################
          w, h = draw.textsize(self.current_lyric)
          font = random.choice(self.fonts)
          # (1280 -350)/2 -350
          len_factor = round((self.master_width/4)/len(self.current_lyric))
          # if len(self.current_lyric) > 50:
          #   splitted = self.current_lyric.split(" ")
          #   words = [ word+'\n' if i == (len(splitted)/2) else word for i, word in enumerate(splitted) ]
          #   self.current_lyric = " ".join(words)

          # lyric_text_x = random.randint(0,40) + (len_factor*25) - int(font['width']*1.2) #(font_factor)    #(self.master_width-w)/2 - 500 +  #random.randint(int(self.master_width/9),int((self.master_width/7)))
          lyric_text_x = random.randint(0,90) + (len_factor*28) - int(font['width']*2.5) #(font_factor)    #(self.master_width-w)/2 - 500 +  #random.randint(int(self.master_width/9),int((self.master_width/7)))
          lyric_text_y = (self.master_height-h)/2 + random.randint(-int(self.master_height/6), int(self.master_height/8)) #random.randint(int(self.master_height/3),2*int(self.master_height/3))
          if lyric_text_x > self.master_width - (self.master_width/7):
            lyric_text_x = self.master_width - lyric_text_x
          if lyric_text_x > self.master_width - (self.master_width/3) and len(self.current_lyric) > 30:
            lyric_text_x = self.master_width - lyric_text_x
        # draw.multiline_text((lyric_text_x, lyric_text_y), self.current_lyric, font=font['font'], fill=self.current_lyric_color, align="center", spacing=30)    
        draw.text((lyric_text_x, lyric_text_y), self.current_lyric, font=font['font'], fill=self.current_lyric_color)    
        # draw.text((lyric_text_x, lyric_text_y), self.current_lyric, font=font, fill=self.current_lyric_color)    
        # use a truetype font  
        # font_futura = ImageFont.truetype("Futura-Book.ttf", 80)  
        # font_papyrus = ImageFont.truetype("PAPYRUS.ttf", 80)  
        # draw.text((lyric_text_x, lyric_text_y), self.current_lyric, font=font_papyrus)  
        # draw.text((10, 300), self.current_lyric, font=font_futura) 
        final_frame = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)  
        # cv.imshow('Fonts', cv_im_processed)  
        ###############################################


      # random_text.put(final_frame)

      if frame_count in peak_frames:
        react_text_x = random.randint(int(self.master_width/3),int((self.master_width/5)*2))
        react_text_y = random.randint(int(self.master_height/3),int(self.master_height/2))
        gray_random = random.randint(1,255)
        react_text_color = (gray_random, gray_random, gray_random)
        react_text_size = random.randint(2,6)
        react_text = 'BOOM' if frame_count%12 > 5 else 'BAP'
        react_text_stroke = random.randint(2,4)
        if self.test_run:
          react_text_size = random.uniform(.5, 1.2)
          react_text_stroke = random.randint(1,3)
        # cv.putText(final_frame, react_text, (react_text_x, react_text_y), cv.FONT_HERSHEY_SIMPLEX, react_text_size, react_text_color, react_text_stroke, cv.LINE_AA)

      final_frame = final_frame[:self.master_height, :self.master_width]
      # final_frame = final_frame[:self.master_height, :self.master_width]

      self.video_writer.write(final_frame) 
      # print(final_frame.shape)

      frame_count += 1
      if frame_count%1000 == 0:
        print(str(int(frame_count))+'/'+str(int(total_video_frames)))

      # cv.imshow('erikiano',final_frame)
      # k = cv.waitKey(33)
      # if k == 27:
      #   break
        
    if self.loaded_video:
      self.loaded_video.release()
    # cv.destroyAllWindows()
    self.destroy_video_writer()
    self.join_audio_video(self.output_name, audio_file_path)

  def edit_music_video(self, audio_file_path):
    anal = AudioAnalyzer()
    print('loading:', audio_file_path)
    self.output_name = audio_file_path.split('/')[-1].split('.')[0] + '.mp4'
    audio_duration, peak_frames = anal.get_peaks_for_audiovisual(audio_file_path, self.fps)
    print('audio duration:',str(audio_duration),'seconds')
    print('audio duration:',str(audio_duration/60),'minutes')
    total_video_frames = int(audio_duration*self.fps)
    print('audio duration:',str(total_video_frames),'frames')
    print('peaks/minutes:',str(int(len(peak_frames)/(audio_duration/60))),'bpm?')

    frame_count = 0
    none_frame_count = 0
    x, y = 100, 200
    # freetype = cv.createFreeType2()
    random_text = RandomText(self.master_width,self.master_height,test_run=self.test_run)
    reversed_frames = 5 # N frames forward then reversed, 5 frames skipped
    reversed_frames = reversed_frames*2
    
    peak_frame_count = 0
    peaks_enabled = False

    print('working on:', self.output_name)
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size, 
                                       self.color_video)

    while frame_count != total_video_frames:
      # self.clock.tick(cycles_per_second)
      if frame_count%self.video_source_rate == 0:
        self.load_next_video()
        
      frame, current_frame = self.get_frame()
      # if frame_count%reversed_frames == 0:
      #   last_frames = []
      # if frame_count%reversed_frames < (reversed_frames/2):
      #   last_frames.append(frame)
      # if frame_count%reversed_frames >= (reversed_frames/2):
      #   reversed_index =(frame_count%reversed_frames) - (len(last_frames)-1)
      #   frame = last_frames[-reversed_index]      
      
      while frame is None:
        none_frame_count += 1
        frame, current_frame = self.get_frame()
        if none_frame_count > 60:
          print('video not working:', self.loaded_videos[self.current_file_index])
          self.load_next_video()
          none_frame_count = 0
      
      if frame_count%self.video_source_rate == 0:
        print(self.loaded_videos[self.current_file_index])
        print(frame.shape)
      
      frame = frame[y:y+self.master_height, x:x+self.master_width]

      if frame_count%int(self.video_source_rate/2) == 0 or frame_count in peak_frames:
        if random.randint(0,10) > 8:
          self.change_current_video_filter()

      if frame_count%15 == 0:
        self.change_current_text()
      if frame_count%10 == 0:
        rand_text_size = random.randint(1,2)
        if self.master_height < 350:
          rand_text_size = 0.5
        # if test_run:
          # random.randint(1,2) #float
        random_text.update(self.current_text,stroke=1,size=random.randint(1,2))

      final_frame = frame
      final_frame = self.apply_video_filters(frame)

      # cv.putText(final_frame, str(frame_count)+self.current_text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (256,256,256))
      # if frame_count % 10 > 5:
        # cv.putText(final_frame, self.current_text, (200, 300), cv.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255))

      final_frame = cv.resize(final_frame, (self.master_width, self.master_height))
      # random_text.put(final_frame)

      if frame_count in peak_frames:
        peaks_enabled = True
        peak_frame_count = 0
        react_text_x = random.randint(int(self.master_width/3),int((self.master_width/5)*2))
        react_text_y = random.randint(int(self.master_height/3),int(self.master_height/2))
        gray_random = random.randint(1,255)
        react_text_color = (gray_random, gray_random, gray_random)
        react_text_size = random.randint(2,6)
        react_text = 'BOOM' if frame_count%12 > 5 else 'BAP'
        react_text_stroke = random.randint(2,4)
        if self.test_run:
          react_text_size = random.uniform(.5, 1.2)
          react_text_stroke = random.randint(1,3)
      if peak_frame_count < 6 and peaks_enabled:
        peak_frame_count += 1 
        # cv.putText(final_frame, react_text, (react_text_x, react_text_y), cv.FONT_HERSHEY_SIMPLEX, react_text_size, react_text_color, react_text_stroke, cv.LINE_AA)
      if peak_frame_count == 6:
        peaks_enabled = False
        peak_frame_count = 0
        if random.randint(0,10) > 6:
          self.change_current_video_filter('none')

      final_frame = final_frame[:self.master_height, :self.master_width]

      self.video_writer.write(final_frame) 
      # print(final_frame.shape)

      frame_count += 1
      if frame_count%1000 == 0:
        print(str(int(frame_count))+'/'+str(int(total_video_frames)))

      # cv.imshow('erikiano',final_frame)
      # k = cv.waitKey(33)
      # if k == 27:
      #   break
        
    if self.loaded_video:
      self.loaded_video.release()
    # cv.destroyAllWindows()
    self.destroy_video_writer()
    self.join_audio_video(self.output_name, audio_file_path)

  def join_audio_video(self, video_file_path, audio_file_path):
    # ffmpeg -i hotpocket.mp4 -i memories/hotpocket.wav -c:v copy -c:a aac hotpocket_audio.mp4
    subprocess.check_call(['ffmpeg','-i',video_file_path,'-i',audio_file_path,'-c:v','copy','-c:a','aac','audio_'+video_file_path])

  def edit_video(self, segments):
    self.video_writer.open('output.mp4', 
                           self.fourcc, 
                           self.fps, 
                           self.master_size, 
                           self.color_video) 
    segment_count = 0
    for file_path in self.all_files:
      valid_file = False

      if '.mov' in file_path.lower() or '.mp4' in file_path.lower():
        self.print_video_segment(file_path, 2)
        valid_file = True

      if '.jpg' in file_path.lower():
        self.print_image_segment(file_path, 2)
        valid_file = True

      if valid_file:
        segment_count += 1
      if segment_count >= segments:
        break
    self.destroy_video_writer()
  

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
  
  def track_object(self):
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space
    buffer = 32 # frames
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    # initialize the list of tracked points, the frame counter,
    # and the coordinate deltas
    pts = deque(maxlen=buffer)
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""
    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
      vs = VideoStream(src=0).start()
    # otherwise, grab a reference to the video file
    else:
      vs = cv.VideoCapture(args["video"])
    # allow the camera or video file to warm up
    time.sleep(2.0)





    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None



    # only proceed if at least one contour was found
    if len(cnts) > 0:
      # find the largest contour in the mask, then use
      # it to compute the minimum enclosing circle and
      # centroid
      c = max(cnts, key=cv.contourArea)
      ((x, y), radius) = cv.minEnclosingCircle(c)
      M = cv.moments(c)
      center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
      # only proceed if the radius meets a minimum size
      if radius > 10:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv.circle(frame, (int(x), int(y)), int(radius),
          (0, 255, 255), 2)
        cv.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)

    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
      # if either of the tracked points are None, ignore
      # them
      if pts[i - 1] is None or pts[i] is None:
        continue
      # check to see if enough points have been accumulated in
      # the buffer
      if counter >= 10 and i == 1 and pts[-10] is not None:
        # compute the difference between the x and y
        # coordinates and re-initialize the direction
        # text variables
        dX = pts[-10][0] - pts[i][0]
        dY = pts[-10][1] - pts[i][1]
        (dirX, dirY) = ("", "")
        # ensure there is significant movement in the
        # x-direction
        if np.abs(dX) > 20:
          dirX = "East" if np.sign(dX) == 1 else "West"
        # ensure there is significant movement in the
        # y-direction
        if np.abs(dY) > 20:
          dirY = "North" if np.sign(dY) == 1 else "South"
        # handle when both directions are non-empty
        if dirX != "" and dirY != "":
          direction = "{}-{}".format(dirY, dirX)
        # otherwise, only one direction is non-empty
        else:
          direction = dirX if dirX != "" else dirY

  # def putRandomText(self, frame):

  def cut_videos_in_segments(self, output_path, seconds=5):
    for p in self.loaded_videos:
      file_name = p.split('/')[-1].split('.')[0]
      print(file_name)
      self.loaded_video = cv.VideoCapture(p)
      total_video_frames = int(self.loaded_video.get(cv.CAP_PROP_FRAME_COUNT))
      frame_count = 0
      written_frame_count = 0
      frame, current_frame = self.get_frame()
      print(current_frame)
      segment_count = 0
      output_name = '{}_{}.mp4'.format(file_name, segment_count)
      segment_frames = seconds * self.fps
      if (total_video_frames/self.fps)/60 > 4:
        segment_frames = 30 * self.fps
      self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                       self.fourcc, 
                                       self.fps, 
                                       (frame.shape[1],frame.shape[0]), 
                                       self.color_video)
      while frame is not None:
        self.video_writer.write(frame) 
        
        if current_frame%segment_frames == 0 and current_frame+(segment_frames/2) < total_video_frames:
          self.destroy_video_writer()
          segment_count += 1
          output_name = '{}_{}.mp4'.format(file_name, segment_count)          
          self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                            self.fourcc, 
                                            self.fps, 
                                            (frame.shape[1],frame.shape[0]), 
                                            self.color_video)
        
        frame, current_frame = self.get_frame()
      self.destroy_video_writer()  
      if self.loaded_video:
        self.loaded_video.release()
  
  def resize_videos_in_segments(self, output_path, seconds=5, zoom=1.35):
    for p in self.loaded_videos:
      file_name = p.split('/')[-1].split('.')[0]
      print(file_name)
      self.loaded_video = cv.VideoCapture(p)
      total_video_frames = int(self.loaded_video.get(cv.CAP_PROP_FRAME_COUNT))
      frame_count = 0
      written_frame_count = 0
      frame, current_frame = self.get_frame()
      print(current_frame)
      segment_count = 0
      output_name = '{}.mp4'.format(file_name, segment_count)
      segment_frames = seconds * self.fps
      if (total_video_frames/self.fps)/60 > 4:
        segment_frames = 30 * self.fps
      self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                       self.fourcc, 
                                       self.fps, 
                                       (frame.shape[1],frame.shape[0]), 
                                       self.color_video)
      f_c = 0
      while frame is not None:
        shape = frame.shape
        if shape[0] > shape[1]:
          ratio = self.master_width/shape[1]
        else:
          ratio = self.master_height/shape[0]
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)
        # rotate the image by 180 degrees
        frame = cv.resize(frame, (self.master_width*3, 3*self.master_height), interpolation=cv.INTER_LINEAR)
        if f_c == 0:
          print(ratio)
          print(frame.shape)
        M = cv.getRotationMatrix2D(center, 0, ratio)
        upscale = cv.warpAffine(frame, M, (w, h))
        (r_h, r_w) = upscale.shape[:2]
        if f_c == 0:
          print(upscale.shape)
        # frame = cv.resize(frame, (r_w, r_h))
        # upscale = upscale[int((r_h-self.master_height)/2):self.master_height+int((r_h+self.master_height)/2),int((r_w-self.master_width)/2):self.master_width+int((r_w+self.master_width)/2)]
        # if f_c == 0:
          # print(upscale.shape)
        frame = cv.resize(upscale, (self.master_width, self.master_height), interpolation=cv.INTER_LINEAR)
        if f_c == 0:
          print(frame.shape)
        self.video_writer.write(frame) 
        
        frame, current_frame = self.get_frame()
        f_c += 1
      self.destroy_video_writer()  
      if self.loaded_video:
        self.loaded_video.release()


      
    

  def play_video(self,filters_enabled=True):
    # self.clock
    self.filters_enabled = filters_enabled
    frame_count = 0
    none_frame_count = 0
    frame_rate = 2
    x, y = 200, 200
    cycles_per_second = 24
    # freetype = cv.createFreeType2()
    text_x = 200
    text_y = 200
    random_text = RandomText(self.master_width,self.master_height)
    random_text_2 = RandomText(self.master_width,self.master_height)
    random_text_3 = RandomText(self.master_width,self.master_height)
    random_text_4 = RandomText(self.master_width,self.master_height)
    random_text_5 = RandomText(self.master_width,self.master_height)
    
    reversed_frames = 10 # N frames forward then reversed, 5 frames skipped
    reversed_frames = reversed_frames*2
    while True:
      self.clock.tick(cycles_per_second)
      if frame_count%85 == 0:
        self.load_next_video()
        
      frame, current_frame = self.get_frame()
      if frame_count%reversed_frames == 0:
        last_frames = []
        # reversed_frames_count = 0
      if frame_count%reversed_frames < (reversed_frames/2):
        last_frames.append(frame)

      if frame_count%reversed_frames >= (reversed_frames/2):
        # 3  2 1 -2 -(len(last_frames)-1)
        # 4  1 2
        # 5  0 3
        # reversed_frames 5
        # -(len(last_frames)-1) -4
        # 0,1,2,3,4
        # 5,6,7,8,9
        # len(last_frames) ((frame_count%reversed_frames)-1)
        reversed_index =(frame_count%reversed_frames) - (len(last_frames)-1)
        # reversed_frames_count += 1
        frame = last_frames[-reversed_index]      
      # else:
        # frame = frame
        # frame, current_frame = previous_frame
        # frame, current_frame = previous_frame
      
      while frame is None:
        none_frame_count += 1
        frame, current_frame = self.get_frame()
        if none_frame_count > 60:
          print('video not working:', self.loaded_videos[self.current_file_index])
          self.load_next_video()
          none_frame_count = 0
      
      # print(frame.shape)
      # if frame.shape[0] #width
      frame = frame[y:y+self.master_height, x:x+self.master_width]

      if frame_count%30 == 0:
        self.change_current_video_filter()
      if frame_count%15 == 0:
        self.change_current_text()
      if frame_count%3 == 0:
        random_text.update(self.current_text)
      if frame_count%4 == 0:
        random_text_2.update(text=None)
        random_text_3.update(text=None)
      if frame_count%5 == 0:
        random_text_4.update()
        random_text_5.update()
      
      final_frame = self.apply_video_filters(frame)

      cv.putText(final_frame, str(frame_count)+self.current_text, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (256,256,256))
      if frame_count % 10 > 5:
        cv.putText(final_frame, self.current_text, (200, 300), cv.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255))
      
     
      # cv.putText(final_frame, self.current_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, text_size , (text_r,text_g,text_b), text_stroke, cv.LINE_AA)
      ###############################################
      # cv_im_rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)  
      # # Pass the image to PIL  
      # pil_im = Image.fromarray(cv_im_rgb)  
      # draw = ImageDraw.Draw(pil_im)  
      # # use a truetype font  
      # font = ImageFont.truetype("PAPYRUS.ttf", 80)  
      # # Draw the text  
      # draw.text((10, 700), text_to_show, font=font)    

      #  # use a truetype font  
      # font_futura = ImageFont.truetype("Futura-Book.ttf", 80)  
      # font_papyrus = ImageFont.truetype("PAPYRUS.ttf", 80)  
      # # Draw the text  
      # draw.text((10, 700), text_to_show, font=font_papyrus)  
      # draw.text((10, 300), text_to_show, font=font_futura) 
      
      # # Get back the image to OpenCV  
      # cv_im_processed = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)  
      # cv.imshow('Fonts', cv_im_processed)  
      ###############################################


      random_text.put(final_frame)
      random_text_2.put(final_frame)
      random_text_3.put(final_frame)
      random_text_4.put(final_frame)
      random_text_5.put(final_frame)
      # cv.FILLED
      # cv.LINE_4
      # cv.LINE_8
      final_frame = cv.resize(final_frame, (self.master_width, self.master_height))
      cv.imshow('erikiano',final_frame)
      # Python: cv.FONT_HERSHEY_SIMPLEX
      # normal size sans-serif font

      # Python: cv.FONT_HERSHEY_PLAIN
      # small size sans-serif font

      # Python: cv.FONT_HERSHEY_DUPLEX
      # normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)

      # Python: cv.FONT_HERSHEY_COMPLEX
      # normal size serif font

      # Python: cv.FONT_HERSHEY_TRIPLEX
      # normal size serif font (more complex than FONT_HERSHEY_COMPLEX)

      # Python: cv.FONT_HERSHEY_COMPLEX_SMALL
      # smaller version of FONT_HERSHEY_COMPLEX

      # Python: cv.FONT_HERSHEY_SCRIPT_SIMPLEX
      # hand-writing style font

      # Python: cv.FONT_HERSHEY_SCRIPT_COMPLEX
      # more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX

      # Python: cv.FONT_ITALIC
      # print('happening', frame_count)
      frame_count += 1


      k = cv.waitKey(33)
      if k == 27:
        break

  def noisy(self,noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      # mean = 0
      # var = 0.2
      # sigma = var**0.5
      mean = 0.5
      var = 0.7
      sigma = var**0.8
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy





  def steal_colors(img, palette_img_src):
      print('# Building replacement palette')
      palette = Palette(palette_img_src, brightness_func=luminosity)

      print('# Subsituting colors')
      img_height = img.size[1]
      list(tqdm(subst_img_colors(img, palette, brightness_func=luminosity), total=img_height))


  def subst_img_colors(img, luminosity2color_palette, brightness_func):
      width, height = img.size
      img = img.load()  # getting PixelAccess
      for j in range(height):
          for i in range(width):
              img[i, j] = luminosity2color_palette[brightness_func(img[i, j])]
          yield 'ROW_COMPLETE' # progress tracking


  def luminosity(pixel):
      r, g, b = pixel[:3]
      return 0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)
    








  def test_filter(self, text, output_path, main_categories=[], seconds=5):
    self.main_categories = main_categories
    self.current_category = main_categories[0]
    all_categories = list(set(main_categories))
    self.load_image_categories(all_categories)
    self.load_fonts(size=100, variation=0.2)
    self.colors = [
      (0,0,0),
      (255,255,255),
      (255,0,0),
    ]
    output_name = 'test_'+text+'.mp4'
    blank_image = np.zeros((self.master_height,self.master_width,3), np.uint8)
    # blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
    # blank_image[:,width//2:width] = (0,255,0)
    # blank_image[:,:] = (0,255,0)
    self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    self.load_next_video()
    self.video_enabled = True
    for i_f in range(seconds*self.fps):
      font = random.choice(self.fonts)
      frame, current_frame = self.get_frame(image=(not self.video_enabled), category=self.current_category)
      shape = frame.shape
      if shape[0] > shape[1]:
        # encontrar ratio de height
        ratio = shape[0]/self.master_width
      else:
        # encontrar ratio de width
        ratio = shape[1]/self.master_height
      frame = cv.resize(frame, (int(self.master_width* ratio) , int(self.master_height*ratio)))
      shape = frame.shape
      # to fit?
      frame = frame[int(shape[0]/2) - int(self.master_height/2):int(shape[0]/2) + int(self.master_height/2), int(shape[1]/2) - int(self.master_width/2):int(shape[1]/2) + int(self.master_width/2)]
      final_frame = cv.resize(frame, (self.master_width, self.master_height))

      # luminosity2color_palette[brightness_func(final_frame[i, j])]
      # max rgb filter


      if i_f%5 == 0:
        frame2, current_frame = self.get_frame(image=True, category=self.current_category)
        shape2 = frame2.shape
        if shape2[0] > shape2[1]:
          # encontrar ratio de height
          ratio = shape2[0]/self.master_width
        else:
          # encontrar ratio de width
          ratio = shape2[1]/self.master_height
        frame2 = cv.resize(frame2, (int(self.master_width* ratio) , int(self.master_height*ratio)))
        shape2 = frame2.shape
        # to fit?
        frame2 = frame2[int(shape2[0]/2) - int(self.master_height/2):int(shape2[0]/2) + int(self.master_height/2), int(shape2[1]/2) - int(self.master_width/2):int(shape2[1]/2) + int(self.master_width/2)]
        final_frame2 = cv.resize(frame2, (self.master_width, self.master_height))

      cv_im_rgb = cv.cvtColor(final_frame,cv.COLOR_BGR2RGB)  

      
      pil_im = Image.fromarray(cv_im_rgb)  
      # pil_im = Image.fromarray(blank_image)  

      draw = ImageDraw.Draw(pil_im)  
      center_text = (self.master_width/4,self.master_height/3)
      # draw.text(center_text, text, font=font['font'], fill=random.choice(self.colors))    
      txt = Image.new('L', (500,50))
      d = ImageDraw.Draw(txt)
      d.text( (0, 0), text,  font=font['font'], fill=255)
      txt_rotate = txt.rotate(17.5,  expand=1)
      pil_im.paste( ImageOps.colorize(txt_rotate, (0,0,0), (255,255,84)), (242,60),  txt_rotate)


      # # Image for text to be rotated
      # img_txt = Image.new('L', font.getsize(rotate_text))
      # draw_txt = ImageDraw.Draw(img_txt)
      # draw_txt.text((0,0), rotate_text, font=font, fill=255)
      # t = img_value_axis.rotate(90, expand=1)




      
      
      final_frame = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)  
      alpha = 0.3
      # [blend_images]
      beta = (1.0 - alpha)
      blended = cv.addWeighted(final_frame, alpha, final_frame2, beta, 0.0)
      # (final_frame*255).astype(np.uint8)
      # final_frame  = cv.cvtColor(final_frame, cv.COLOR_HSV2BGR)
      # final_frame = self.noisy('gauss',final_frame)
      final_frame = np.uint8(blended)


      self.video_writer.write(final_frame) 
    self.loaded_video.release()
    self.destroy_video_writer()  

  def create_logo_sprite(self, text, output_path, seconds=5):
    self.load_fonts(size=100, variation=0)
    self.colors = [
      (0,0,0),
      (255,255,255),
      (255,0,0),
    ]
    output_name = text+'_text_sprite.mp4'
    blank_image = np.zeros((self.master_height,self.master_width,3), np.uint8)
    # blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
    # blank_image[:,width//2:width] = (0,255,0)
    blank_image[:,:] = (0,255,0)
    self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    for i_f in range(seconds*self.fps):
      font = random.choice(self.fonts)
      pil_im = Image.fromarray(blank_image)  
      draw = ImageDraw.Draw(pil_im)  
      center_text = ((self.master_width/2)-(2*font['width']),self.master_height/3)
      draw.text(center_text, text, font=font['font'], fill=random.choice(self.colors))    
      final_frame = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)  
      self.video_writer.write(final_frame) 
    self.destroy_video_writer()  
  
  def apply_image_filters(self,f):
    frame = None
    if self.current_image_filter == 'none':
      frame = f
    elif self.current_image_filter == 'invert':
      frame = (255-f)
    elif self.current_image_filter == 'change-colors':
      lower = (0, 0, 0) # lower bound for each channel
      upper = (25, 25, 25) # upper bound for each channel
      # create the mask and use it to change the colors
      dark_mask = cv.inRange(f, lower, upper)
      lower = (230, 230, 230)
      upper = (255, 255, 255)
      light_mask = cv.inRange(f, lower, upper)
      # r_g_b = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
      frame =  f
      colors = list(self.colors)
      if self.current_lyric_color:
        lyric_color = self.current_lyric_color
        colors.remove((lyric_color[2],lyric_color[1],lyric_color[0]))
      frame[dark_mask != 0] = random.choice(colors)
      frame[light_mask != 0] = random.choice(colors)
    return frame

  def change_current_image_filter(self):
    filters = list(self.image_filters)
    filters.remove(self.current_image_filter)
    self.current_image_filter = random.choice(filters)

  def change_current_video_filter(self, filter_name=None):
    if filter_name:
      self.current_video_filter = filter_name
    else:
      filters = list(self.video_filters)
      filters.remove(self.current_video_filter)
      shuffle(filters)
      self.current_video_filter = filters[0]
  
  def change_current_text(self):
    # words = ['erikiado', 'erikiano', 'programacion', 'hello world', 'yuxtaposition', '@@@@@@']
    words = ['erikiado', 'erikiano', '@@@@@@', '######', '%%%%%%%', 'sube','baja','detente','sigue','escuchame','callate','sigueme','vete','explora']
    next_index = random.randint(0,len(words)-1)
    self.current_text = words[next_index]
  
  def apply_video_filters(self,f):
    frame = None
    mask = None
    # self.current_video_filter = 'backsub-color-invert'
    if self.current_video_filter == 'none' or not self.filters_enabled:
      frame = f
    elif self.current_video_filter == 'backsub-color-invert':
      mask = self.backSub.apply(f)
      mask = 255 - mask
      frame = cv.bitwise_and(f,f,mask = mask)
      # frame = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
    elif self.current_video_filter == 'backsub-color':
      mask = self.backSub.apply(f)
      frame = cv.bitwise_and(f,f,mask = mask)
    elif self.current_video_filter == 'backsub-color-color':
      mask = self.backSub.apply(f)
      # color_mask = 255 - mask
      # print(color_mask)
      frame = cv.bitwise_and(f,f,mask = mask)
      r_g_b = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
      frame[np.where((frame==[0,0,0]).all(axis=2))] = r_g_b
    elif self.current_video_filter == 'backsub':
      mask = self.backSub.apply(f)
      frame = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)
    elif self.current_video_filter == 'invert':
      frame = (255-f)
    return frame

  def print_video_segment(self, path, seconds=0, duration_frames=300):
    frame_count = 0
    current_seconds = 0
    none_frame_count = 0
    dd = cv.VideoCapture(path)
    x, y = 200, 200
    while True:
      ret, frame = dd.read()
      if frame is None:
          none_frame_count += 1
          if none_frame_count > 100:
            break
          continue
      else:
        frame_count += 1
        frame = frame[y:y+self.master_height, x:x+self.master_width]
        if frame_count%30 == 0:
          self.change_current_video_filter()
        final_frame = self.apply_video_filters(frame)
        if seconds > 0:
          current_seconds = frame_count/self.fps
          if current_seconds >= seconds:
            break
        elif frame_count > duration_frames:
            break
        self.video_writer.write(final_frame) 
        self.total_frames += 1
        
    # print(none_frame_count)
    dd.release()

  def print_image_segment(self, path, seconds=0, duration_frames=100):
    frame_count = 0
    current_seconds = 0
    none_frame_count = 0
    image = cv.imread(path)
    image_height = image.shape[0]
    image_width = image.shape[1]
    x, y = 100, 200
    while frame_count < duration_frames:
      frame_count += 1
      self.total_frames += 1

      cut_y = y+self.master_height
      if cut_y+frame_count <= image_height: 
        y += frame_count
        cut_y += frame_count
      cut_x = x+self.master_width
      if cut_x+frame_count <= image_width: 
        x += frame_count
        cut_x += frame_count
      # else:

      frame = image[y:cut_y, x:cut_x]
      if frame_count%30 == 0:
        self.change_current_image_filter()
      final_frame = self.apply_image_filters(frame)
      # print('image',final_frame)
      self.video_writer.write(final_frame) 
      if none_frame_count > 50:
        break
      if seconds > 0:
        current_seconds = frame_count/self.fps
        if current_seconds >= seconds:
          break


# video_editor = VideoEditor(
#   file_paths = [
#     '/Users/erikiado/Code/emily/memories/',
#     '/Volumes/SSD1/visual/textures/',
#     '/Volumes/SSD1/visual/japan2019/02kyoto'
#   ],
#   test_run=False)
# video_editor.edit_music_video('/Users/erikiado/Code/emily/memories/acercate.wav')

# video_editor.edit_video(6)
# video_editor.edit_music_video('/Users/erikiado/Code/emily/memories/labbin.wav')
# video_editor.edit_music_video('/Users/erikiado/Code/emily/memories/hotpocket.wav')
# video_editor.play_video(filters_enabled=False)


# video_editor = VideoEditor(
#   file_paths = [
#     '/Volumes/SSD1/visual/japan2019/00ueno'
#   ],
#   test_run=False)
# video_editor.edit_music_video('/Users/erikiado/Music/podcast/01saliendodeuenopark.wav')

# podcast
# video_editor = VideoEditor(
#   file_paths = [
#     '/Volumes/SSD1/visual/textures/',
#     '/Volumes/SSD1/visual/hablando/',
#     '/Volumes/SSD1/visual/japan2019/00ueno',
#     '/Volumes/SSD1/visual/japan2019/01shinjuku',
#     '/Volumes/SSD1/visual/japan2019/02kyoto'
#   ],
#   test_run=False,
#   # fps=30,
#   video_source_rate=30*8,
#   height=500, width=500, )
# video_editor.edit_music_video('/Users/erikiado/Music/podcast/03unbarriokoreanoentokyo.wav')

editor_configuration = dict(
  file_paths=[
    # '/Users/erikiado/Code/emily/cut_videos/',
    '/Users/erikiado/Code/emily/alpacino/',
    # '/Users/erikiado/Code/emily/test_session/',

  ],
  fonts_path='/Users/erikiado/Code/emily/fonts/',
  test_run=False,
  # fps=30,
  image_source_rate=3,
  # video_source_rate=30*8,
  height=720, width=1280,
  # height=1080, width=1920,
  fps=30
)

video_editor = VideoEditor(**editor_configuration)
# video_editor.edit_music_image_gallery('/Users/erikiado/Music/erikiado/feeldacreep.wav')
# video_editor.edit_music_image_gallery('/Users/erikiado/Code/emily/tury_al_pacino/alpacino.wav', main_categories=['al pacino'], peak_categories=['crazy','scarface'])
      # if frame_count == int(total_video_frames/6):
      #   self.peak_duration = 32
      #   self.image_source_rate = 4
      # if frame_count == 2*int(total_video_frames/6):
      #   self.peak_duration = 20
      #   self.image_source_rate = 2
      # if frame_count == 3*int(total_video_frames/6):
      #   self.peak_duration = 12
      #   self.image_source_rate = 6
      # if frame_count == 4*int(total_video_frames/6):
      #   self.peak_duration = 20
      #   self.image_source_rate = 2
      # if frame_count == 5*int(total_video_frames/6):
      #   self.peak_duration = 32
      #   self.image_source_rate = 4
# video_editor.test_filter('test_name','/Users/erikiado/Code/emily/', main_categories=['dark'])
# video_editor.create_logo_sprite('erikiado64','/Users/erikiado/Code/emily/')
video_editor.edit_music_image_gallery('/Users/erikiado/Code/emily/tury_al_pacino/alpacino.wav', 
                                      lyrics='/Users/erikiado/Code/emily/alpacino/lyrics.txt',
                                      main_categories=['0','nocturnal','recurrent','recurrent'], 
                                      peak_categories=['moon'])


# video_editor.create_logo_sprite('nu9ve','/Users/erikiado/Code/emily/')
# video_editor.create_logo_sprite('tury fresh','/Users/erikiado/Code/emily/')
# video_editor.edit_music_image_gallery('/Users/erikiado/Music/podcast/03unbarriokoreanoentokyo.wav')
# video_editor.cut_videos_in_segments('/Users/erikiado/Code/emily/cut_videos/results')
# video_editor.resize_videos_in_segments('/Users/erikiado/Code/emily/cut_videos/results')






# # check if image is transparent
# for entry in `pwd`/test_session/*           
# do
#   echo "$entry"; convert $entry -format "%[opaque]" info:
# done





# >>> import glob
# >>> glob.glob('./[0-9].*')
# ['./1.gif', './2.txt']
# >>> glob.glob('*.gif')
# ['1.gif', 'card.gif']
# >>> glob.glob('?.gif')
# ['1.gif']





























# self.vout = None
# out = cv.VideoWriter('project.mp4',cv.VideoWriter_fourcc(*'MP42'), 24, MASTER_SIZE)

# paste 3 segments
# normal video
# background substraction segment video
# slowed video
# normal video



# data mapper to video

# screen recording

# template recording

# effect on video

# videos to master

# effects on master video





# exporting video

# img_array = []
# for filename in glob.glob('C:/New folder/Images/*.jpg'):
#     img = cv.imread(filename)
#     height, width, layers = img.shape
#     MASTER_SIZE = (width,height)
#     img_array.append(img)
 
 
# out = cv.VideoWriter('project.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, MASTER_SIZE)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()


# saving as mp4
# self._name = name + '.mp4'
# self._cap = VideoCapture(0)
# self._fourcc = VideoWriter_fourcc(*'MP4V')
# self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))





    # # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # # fgMask = backSub.apply(frame)
    # # cv.imshow('FG Mask', fgMask)

    # final_frame = apply_filters(frame, frame_count)
    
    # # segment.append(final_frame)
    # vout.write(final_frame) 
    # if frame_count > 300:
    #   break

    # # cv.putText(final_frame, str(frame_count)+' - '+str(dd.get(cv.CAP_PROP_POS_FRAMES)), (200, 200), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,0))
    # # cv.imshow('Frame', final_frame)
    