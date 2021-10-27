from math import fabs
import time
import datetime
import subprocess
import random
import logging as log
import pickle

from os import listdir
from os.path import isfile, join as join_paths
from pathlib import Path
from random import shuffle
from itertools import count
from collections import deque

import imutils
import pygame
import ffmpeg
import cv2 as cv
import numpy as np
from scipy import ndimage
from PIL import ImageFont, ImageDraw, Image, ImageOps, ImageFilter

from audio.analysis.audio_analyzer import AudioAnalyzer
from visual.video.frame_queue import FrameQueue
from visual.text.fonts import FONT_WIDTHS
from utils.constants import PATHS, DEFAULT_FPS
from utils.time_utils import timestamp_to_video_frame


class VideoEditor:
  frame_queue = None
  video_writer = None
  clock = None
  paths = []
  current_text = None
  current_lyric = None
  text_sprite = False
  text_sprite_font_rate = 5
  lyrics_enabled = False
  current_image_frame = 0
  lyrics = []
  loaded_fonts = []
  loaded_wavs = []
  fonts = []
  color_video = True
  total_video_frames = 0
  master_height = 1000
  master_width = 1000
  master_size = (master_width, master_height)
  current_lyric_color = None
  filters_enabled = True  
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
  selected_filters = []
  freeze_filter = False
  current_video_filter = 'none'
  current_image_filter = 'none'

  current_category = None
  main_categories = []
  peak_categories = []
  current_categories = []
  current_peak_categories = []
  fonts_path = None

  live_video = True
  background = False
  peak_frames = []
  verbose = False
  debug = False
  log_level = None
  frame_count = None
  clips = []
  lyric_font = None
  lyric_position = None
  output_name = None
  last_font = None 

  def __init__(self, file_paths, output_path=None, fonts_path=None, height=None, width=None, fps=None, q=None, verbose=False, debug=False, init_fonts=False):
    self.fps = fps if fps else DEFAULT_FPS
    self.verbose = verbose if verbose else False
    self.debug = debug if debug else False
    self.log_level = None
    if debug:
      self.log_level = log.DEBUG # all; develop
    elif verbose:
      self.log_level = log.INFO # edit mode; file errors for cleanup
    else:
      self.log_level = log.ERROR # play mode

    log.basicConfig(format='[%(asctime)s](%(levelname).1s) %(message)s', level=self.log_level)# datefmt='%d-%b-%y %H:%M:%S')
    # log.error('%(name)s raised an error')
    # try:
    #   c = a / b
    # except Exception as e:
    #   log.exception("Exception occurred") # error on try that logs traceback
    self.output_path = output_path if output_path else os.path.abspath(os.getcwd())
    self.log('progress', 'output directory: {}'.format(self.output_path))
    if q:
      if q == '1080':
        self.master_height = 1080
        self.master_width = 1920
      elif q == '720':
        self.master_height = 720
        self.master_width = 1280
      elif q == 'test':
        self.master_height = 300
        self.master_width = 300
    else: 
      if height and width:
        self.master_height = height
        self.master_width = width
      elif height:
        self.master_height = height
        self.master_width = height
      elif width:
        self.master_height = width
        self.master_width = width
      else:
        self.master_height = 720
        self.master_width = 1280   
    self.master_size = (self.master_width, self.master_height)

    self.frame_queue = FrameQueue(file_paths, 
                                  height=self.master_height, 
                                  width=self.master_width, 
                                  image_source_rate=3, 
                                  video_source_rate=self.fps*8, 
                                  fps=self.fps)
    self.paths += file_paths
    self.get_file_paths()
    self.video_writer = None
    self.fourcc = cv.VideoWriter_fourcc(*'mp4v')
    self.backSub = cv.createBackgroundSubtractorMOG2()
    # self.backSub = cv.createBackgroundSubtractorKNN()
    self.clock = pygame.time.Clock()
    self.current_text = 'erikiano'
    self.filters_enabled = True
    
    if fonts_path and init_fonts:
      self.fonts_path = fonts_path
      self.load_fonts()


  def destroy_video_writer(self):
    self.video_writer.release()


  def log(self, level, message):
    if level == 'progress':
      log.info(message)
    elif level == 'info':
      log.info(message)
    elif level == 'asset' and self.verbose:
      log.info(message)
    elif level == 'structure' and self.verbose:
      log.info(message)
    elif level == 'warning':
      log.warning(message)
    elif level == 'error':
      log.error(message)
    elif level == 'debug':
      log.debug(message)


  def generate_output_name(self, name, directory=None):
    if directory:
      Path(join_paths(self.output_path, directory)).mkdir(parents=True, exist_ok=True)
    else:
      Path(self.output_path).mkdir(parents=True, exist_ok=True)
    name += '_'+str(self.master_width)+'x'+str(self.master_height)+'.mp4' 
      # only supporting mp4; update VideoWriter
    return join_paths(self.output_path, directory, name)


  # ASSET MANAGEMENT

  def load_fonts(self, size=50, variation=0.1, word=None):
    full_height = True
    txt_img = Image.new('RGBA', (self.master_width, self.master_height), (255,255,255,0))
    txt_draw = ImageDraw.Draw(txt_img)

    # txt = "erikiado64"
    txt = "debuggeando"
    # txt = "tury fresh"
    if word:
      txt = word

    self.loaded_fonts = [join_paths(self.fonts_path, f) for f in listdir(self.fonts_path) if isfile(join_paths(self.fonts_path, f)) and 'tf' in f ]
    self.log('asset', 'loaded fonts: {}'.format([fp.split('/')[-1].split('.')[0] for fp in self.loaded_fonts ]))
    try:
      font_size_map = pickle.load(open(PATHS['font_size_map'], 'rb'))
      self.log('asset', 'loaded font size map')
    except (OSError, IOError) as e:
      font_size_map = dict()
      self.log('warning', 'font size map not loaded')
    for p in self.loaded_fonts:
      # random varied size
      font_name = p.split('/')[-1].split('.')[0]

      # big = size-int((size*variation)-(FONT_WIDTHS[font_name]*(size//10)))
      # small = size+int(size*variation)
      # if small > big:
      #   tmp_small = big
      #   big = small
      #   small = tmp_small
      # font_size = random.randint(small, big)
      # font = ImageFont.truetype(p, font_size)
      
      size_key = '{}x{}'.format(self.master_height, self.master_width)
      if size_key not in font_size_map:
        font_size_map[size_key] = dict()
      if font_name not in font_size_map[size_key]:
        font_size_map[size_key][font_name] = dict()

      # if (full_height and 
      #     'height' not in font_size_map[size_key][font_name]):
      #   print('loading height font size map')
      #   fontsize = int(3*(self.master_height/4))  # starting font size
      #   img_fraction = 1 # portion of image width you want text width to be
      #   font = ImageFont.truetype(p, fontsize)
      #   h = txt_draw.textsize(txt, font=font)[1]
      #   while h < img_fraction*txt_img.size[1]:
      #     # iterate until the text size is just larger than the criteria
      #     # if h+(loquemideelquesigue?)< img_fraction*txt_img.size[1]:
      #     fontsize += 10
      #     font = ImageFont.truetype(p, fontsize)
      #     h = txt_draw.textsize(txt, font=font)[1]
      #   # while h < img_fraction*txt_img.size[1]:
      #   #   fontsize += 1
      #   self.log('asset', str(fontsize)+' - '+p)
      #   font_size_map[size_key][font_name]['height'] = fontsize

      # if (full_height and 
      #     'width' not in font_size_map[size_key][font_name]):
      #   print('loading width font size map')
      #   fontsize = 20  # starting font size
      #   # fontsize = int(2*(self.master_width/4))  # starting font size
      #   img_fraction = 1 # portion of image width you want text width to be
      #   font = ImageFont.truetype(p, fontsize)
      #   w = txt_draw.textsize(txt, font=font)[0]
      #   while w < img_fraction*txt_img.size[0]:
      #     # iterate until the text size is just larger than the criteria
      #     # if h+(loquemideelquesigue?)< img_fraction*txt_img.size[1]:
      #     fontsize += 10
      #     font = ImageFont.truetype(p, fontsize)
      #     w = txt_draw.textsize(txt, font=font)[0]
      #   # while h < img_fraction*txt_img.size[1]:
      #   #   fontsize += 1
      #   self.log('asset', str(fontsize)+' - '+p)
      #   font_size_map[size_key][font_name]['width'] = fontsize

      fontsize = 20  # starting font size
      if (full_height and 
          'word' not in font_size_map[size_key][font_name]):
        print('loading word font size map')
        fontsize = 20  # starting font size
        # fontsize = int(2*(self.master_width/4))  # starting font size
        img_fraction = 0.8 # portion of image width you want text width to be
        font = ImageFont.truetype(p, fontsize)
        tsize = txt_draw.textsize(txt, font=font)
        w = tsize[0]
        h = tsize[1]
        while w < int(img_fraction*txt_img.size[0]) and h < int(img_fraction*txt_img.size[1]):
          # iterate until the text size is just larger than the criteria
          # if h+(loquemideelquesigue?)< img_fraction*txt_img.size[1]:
          fontsize += 5
          font = ImageFont.truetype(p, fontsize)
          tsize = txt_draw.textsize(txt, font=font)
          w = tsize[0]
          h = tsize[1]
        # while h < img_fraction*txt_img.size[1]:
        #   fontsize += 1
        self.log('asset', str(fontsize)+' - '+p)
        font_size_map[size_key][font_name]['width'] = fontsize

      font_width = FONT_WIDTHS[font_name] * (0.8 * fontsize)

      self.fonts.append(dict(font=font, width=font_width))
    pickle.dump(font_size_map, open(PATHS['font_size_map'], 'wb'))



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
      only_wavs = list(filter(lambda f: '.wav' in f.lower(), only_files))
      self.loaded_wavs += only_wavs
      # self.loaded_images += only_images
      # self.loaded_videos += only_videos
      # self.all_files = self.loaded_images + self.loaded_videos
      self.all_files = self.loaded_wavs
      shuffle(self.all_files)


  # TEXT FUNCTIONS
  def parse_lyrics(self, lyrics):
    if lyrics:
      self.lyrics = []
      self.log('progress', 'parsing lyrics: {}'.format(lyrics))
      with open(lyrics, 'r') as f:
        for l in f.readlines():
          if '(' in l  and ')' in l and '-' in l:
            lyric_line = dict(start=timestamp_to_video_frame(self.fps, l[1:10]),end=timestamp_to_video_frame(self.fps, l[11:20]),text=l[22:].rstrip())
            self.lyrics.append(lyric_line)


  # AUDIO FUNCTIONS

  def get_audio_file_path(self, file_name):
    for p in self.all_files:
      if file_name in p:
        return p
    return False

  def get_audio_metadata(self, audio_file_path):
    audio_analyzer = AudioAnalyzer()
    audio_duration, peak_frames = audio_analyzer.get_audio_metadata(audio_file_path, self.fps)
    total_video_frames = int(audio_duration*self.fps)
    self.log('info', 'audio duration: {} seconds'.format(str(audio_duration)))
    self.log('info', 'audio duration: {} minutes'.format(str(audio_duration/60)))
    self.log('info', 'audio duration: {} frames'.format(str(total_video_frames)))
    self.log('info', 'peaks/minutes:  {} bpm?'.format(str(int(len(peak_frames)/(audio_duration/60)))))
    return total_video_frames, peak_frames

  def join_audio_video(self, video_file_path, audio_file_path):
    # ffmpeg -i hotpocket.mp4 -i memories/hotpocket.wav -c:v copy -c:a aac hotpocket_audio.mp4
    path_parts = video_file_path.split('.')
    path_parts[-2] += '_audio'
    final_video_path = '.'.join(path_parts)
    subprocess.check_call(['ffmpeg','-i',video_file_path,'-i',audio_file_path,'-c:v','copy','-c:a','aac', final_video_path])

  def join_timed_audio_video(self, video_file_path, source_video_file, start_time=0, duration=10):
    # strip ranged audio
    # ffmpeg -i hotpocket.mp4 -c:v copy -c:a aac hotpocket_audio.mp4
    clip_audio_path = video_file_path.replace('.mp4','_a.mp4')
    subprocess.check_call(['ffmpeg','-ss',str(start_time),'-t',str(duration),'-i',source_video_file,'-c:v','copy','-c:a','aac', clip_audio_path])
    # audio_file_path = new file
    # ffmpeg -i hotpocket.mp4 -i memories/hotpocket.wav -c:v copy -c:a aac hotpocket_audio.mp4

    # path_parts = video_file_path.split('.')
    # path_parts[-2] += '_audio'
    # final_video_path = '.'.join(path_parts)
    # subprocess.check_call(['ffmpeg','-i',video_file_path,'-i',audio_file_path,'-c:v','copy','-c:a','aac', final_video_path])


  # MAIN LOOP MODES
  def play_live_video(self, clips=[], colors=None, filters=None):
    self.clips = clips
    if colors:
      self.colors = colors
    self.video_enabled = True
    # self.filters_enabled = False
    if filters:
      self.filters_enabled = True
      self.freeze_filter = True
      self.current_image_filter = 'invert'
      self.current_image_filter = 'sort'
      self.selected_filters = ['none','invert']
      # 'none','invert','backsub','backsub-color','backsub-color-invert','backsub-color-color'
    else:
      self.filters_enabled = False


    self.fps = 20

    self.video_source_rate = self.fps * 1
    self.image_source_rate = self.fps * 1
    self.source_rate = self.fps * 1
    self.main_video_loop(live_video=True)


  def generate_lyric_video(self, audio_file_path, main_categories=[], peak_categories=[], lyrics=None, clips=[], colors=[], font_colors=[]):
    self.main_categories = main_categories
    self.peak_categories = peak_categories
    self.clips = clips
    if colors:
      self.colors = [ (c[2],c[1],c[0]) for c in colors ] # bgr for opencv
      if font_colors:
        self.font_colors = font_colors
      else:
        self.font_colors = colors
    else:
      self.colors = []
      self.font_colors = []
    audio_name = audio_file_path.split('/')[-1].split('.')[0]
    self.total_video_frames, self.peak_frames = self.get_audio_metadata(audio_file_path)
    self.output_name = self.generate_output_name(audio_name, 'lyric_videos')
    self.parse_lyrics(lyrics)
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size, 
                                       self.color_video)
    self.video_enabled = True
    source_rate = 20
    image_source_rate = 5
    self.frame_queue.update_source_rate(source_rate, image_source_rate=image_source_rate)
    self.main_video_loop(live_video=False)
    self.join_audio_video(self.output_name, audio_file_path)


  def test_filter(self, test_name, filters=None, seconds=5):
    test_name = 'test_' + test_name
    self.output_name = self.generate_output_name(test_name, 'filter_tests')
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    self.video_enabled = True
    if filters:
      self.selected_filters = filters
    else:
      self.selected_filters = ['invert']
    self.change_current_image_filter(self.selected_filters[0])
    self.total_video_frames = seconds*self.fps
    self.main_video_loop(live_video=False)


  # MAIN LOOP FUNCTIONS
  def react_to_peak(self, frame_count):
    if frame_count in self.peak_frames:
      self.peaks_enabled = True
      self.peak_frame_count = 0
      self.current_category = random.choice(self.current_peak_categories) if len(self.current_peak_categories) else None
    if self.peak_frame_count < self.peak_duration and self.peaks_enabled:
      self.peak_frame_count += 1 
    if self.peak_frame_count == self.peak_duration:
      self.peaks_enabled = False
      self.current_category = random.choice(self.current_categories) if len(self.current_categories) else None
      self.peak_frame_count = 0
      if random.randint(0, 10) > 6:
        self.change_current_video_filter('none')


  def update_episode_options(self, frame_count):
    if frame_count in self.episode_starts:
      for i, e in enumerate(self.clips):
        if self.episode_starts[i] == frame_count:
          options = e['options']
          # self.frame_queue.update()
          if 'peak_duration' in options:
            self.peak_duration = options['peak_duration']
          if 'image_source_rate' in options:
            self.image_source_rate = options['image_source_rate']
          if 'video_source_rate' in options:
            self.video_source_rate = options['video_source_rate']
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


  def update_current_lyrics(self, frame_count):
    if frame_count in self.lyric_ends:
      for i, l in enumerate(self.lyrics):
        if self.lyric_ends[i] == frame_count:
          self.lyrics_enabled = False
    if frame_count in self.lyric_starts:
      for i, l in enumerate(self.lyrics):
        if self.lyric_starts[i] == frame_count:
          self.lyrics_enabled = True
          self.current_lyric = l['text']
    # change centered text?
    if frame_count%15 == 0:
        self.change_current_text()


  def dropShadow(self, image, offset=(5,5), background=0xffffff, shadow=0x444444, 
                border=8, iterations=3):
    """
    Add a gaussian blur drop shadow to an image.  
    
    image       - The image to overlay on top of the shadow.
    offset      - Offset of the shadow from the image as an (x,y) tuple.  Can be
                  positive or negative.
    background  - Background colour behind the image.
    shadow      - Shadow colour (darkness).
    border      - Width of the border around the image.  This must be wide
                  enough to account for the blurring of the shadow.
    iterations  - Number of times to apply the filter.  More iterations 
                  produce a more blurred shadow, but increase processing time.
    """
    
    # Create the backdrop image -- a box in the background colour with a 
    # shadow on it.
    totalWidth = image.size[0] + abs(offset[0]) + 2*border
    totalHeight = image.size[1] + abs(offset[1]) + 2*border
    back = Image.new(image.mode, (totalWidth, totalHeight), background)
    
    # Place the shadow, taking into account the offset from the image
    shadowLeft = border + max(offset[0], 0)
    shadowTop = border + max(offset[1], 0)
    back.paste(shadow, [shadowLeft, shadowTop, shadowLeft + image.size[0], 
      shadowTop + image.size[1]] )
    
    # Apply the filter to blur the edges of the shadow.  Since a small kernel
    # is used, the filter must be applied repeatedly to get a decent blur.
    n = 0
    while n < iterations:
      back = back.filter(ImageFilter.BLUR)
      n += 1
      
    # Paste the input image onto the shadow backdrop  
    imageLeft = border - min(offset[0], 0)
    imageTop = border - min(offset[1], 0)
    back.paste(image, (imageLeft, imageTop))
    
    return back


  def apply_text(self, image_draw, text, font, offset=(0,0), alpha=255):
    # rotate texts
    # draw = ImageDraw.Draw(pil_im)  
    # txt = Image.new('L', (500,50))
    # d = ImageDraw.Draw(txt)
    # d.text( (0, 0), text,  font=font['font'], fill=255)
    # txt_rotate = txt.rotate(17.5,  expand=1)
    # pil_im.paste( ImageOps.colorize(txt_rotate, (0,0,0), (255,255,84)), (242,60),  txt_rotate)
    position = 'center' # left, right, top_left, top, top_right
    # center
    width_text, height_text = image_draw.textsize(text, font=font['font'])
    text_position = (((self.master_width-width_text)/2)+offset[0],((self.master_height-height_text)/2)+offset[1])
    img_text = image_draw.text(text_position, 
                    text, 
                    font=font['font'], 
                    fill=(*random.choice(self.colors), alpha))
    # self.dropShadow(img_text)

  def apply_texts(self, final_frame, frame_count):
    text_enabled = False
    if self.text_sprite or self.lyrics_enabled:
      text_enabled = True
      cv_im_rgb = cv.cvtColor(final_frame,cv.COLOR_BGR2RGB)  
      pil_im = Image.fromarray(cv_im_rgb).convert('RGBA')
      draw = ImageDraw.Draw(pil_im)
      # same size transparent image for texts
      txt_img = Image.new('RGBA', pil_im.size, (255,255,255,0))
      txt_draw = ImageDraw.Draw(txt_img)

    if self.text_sprite:
      if frame_count%self.text_sprite_font_rate == 0:
        self.last_font = random.choice(self.fonts)
      # self.apply_text(draw, self.text_sprite, self.last_font)
      self.apply_text(txt_draw, self.text_sprite, self.last_font, alpha=100)
      n = 0
      iterations = 10
      while n < iterations:
        back = txt_img.filter(ImageFilter.BLUR)
        n += 1
      self.apply_text(txt_draw, self.text_sprite, self.last_font, offset=(5,5))
      # self.apply_text(txt_draw, self.text_sprite, self.last_font, offset=(10,10), alpha=100)
    if self.lyrics_enabled:
      if frame_count%self.lyric_text_rate == 0 or not self.current_lyric_color or (self.global_peaks_enabled and self.peaks_enabled and frame_count%self.peak_lyric_text_rate==0):
        # previous_lyric_frame = 
        self.current_lyric_color = random.choice(self.font_colors)
        lyric_text_size = random.uniform(.5, 1.2)
        lyric_text_stroke = random.randint(1,2)
        w, h = draw.textsize(self.current_lyric)
        self.lyric_font = random.choice(self.fonts)
        len_factor = round((self.master_width/4)/len(self.current_lyric))
        lyric_text_x = random.randint(0,90) + (len_factor*28) - int(self.lyric_font['width']*2.5) #(font_factor)    #(self.master_width-w)/2 - 500 +  #random.randint(int(self.master_width/9),int((self.master_width/7)))
        lyric_text_y = (self.master_height-h)/2 + random.randint(-int(self.master_height/6), int(self.master_height/8)) #random.randint(int(self.master_height/3),2*int(self.master_height/3))
        if lyric_text_x > self.master_width - (self.master_width/7):
          lyric_text_x = self.master_width - lyric_text_x
        if lyric_text_x > self.master_width - (self.master_width/3) and len(self.current_lyric) > 30:
          lyric_text_x = self.master_width - lyric_text_x
        self.lyric_position = (lyric_text_x, lyric_text_y)
      draw.text(self.lyric_position, self.current_lyric, font=self.lyric_font['font'], fill=self.current_lyric_color)

    if text_enabled:
      combined = Image.alpha_composite(pil_im, txt_img)
      final_frame = cv.cvtColor(np.array(combined), cv.COLOR_RGB2BGR)  
    return final_frame


  def resize_frame_for_output(self, frame):
    # resize image
    shape = frame.shape
    # if width larger resize height until it fits output frame
    if shape[1] > shape[0]:
      ratio = self.master_height/shape[0]
    else:
      ratio = self.master_width/shape[1]
    frame = cv.resize(frame, (int(shape[1]*ratio),int(shape[0]*ratio)))
    shape = frame.shape
    # crop center of image, check exact bound, larger bound should be centered
    frame = frame[int(shape[0]/2) - int(self.master_height/2):int(shape[0]/2) + int(self.master_height/2), int(shape[1]/2) - int(self.master_width/2):int(shape[1]/2) + int(self.master_width/2)]
    # resize to fit output
    final_frame = cv.resize(frame, (self.master_width, self.master_height))
    return final_frame

  def parse_episodes_categories(self):
    episodes_categories = []
    for e in self.clips:
      if 'categories' in e['options']:
        episodes_categories += e['options']['categories']
      if 'peak_categories' in e['options']:
        episodes_categories += e['options']['peak_categories']
    return episodes_categories


  def apply_filters(self, f, frame_count):
    if self.filters_enabled:
      # randomly 20% check if peak frame or source rate to change image filter
      if not self.freeze_filter and random.randint(0,10) > 8:
        if frame_count%int(self.image_source_rate/2) == 0 or frame_count in self.peak_frames:
          self.change_current_image_filter()
        if frame_count%int(self.video_source_rate/2) == 0 or frame_count in self.peak_frames:
          self.change_current_video_filter()
      # peak order?

    if self.filters_enabled:
      # final_frame = self.apply_video_filters(f)
      final_frame = self.apply_image_filters(f)
      return final_frame
    return f

  def sigmoid(x):
    return 1/(1+math.exp(-x))

  def apply_image_filters(self,f):
    frame = None
    if self.current_image_filter == 'none':
      frame = f
    elif self.current_image_filter == 'invert':
      frame = (255-f)
    elif self.current_image_filter == 'half-sort':

      lf = ndimage.gaussian_laplace(f, sigma=0.001) # colored lsd
      y1 = 0
      y2 = 1080
      x1 = 0
      x2 = 820
      f[y1:y2, x1:x2] = lf[y1:y2, x1:x2]
      frame = f
    elif self.current_image_filter == 'sort':
      # frame = f > f.mean()

      # sx = ndimage.sobel(f, axis=0, mode='wrap') # constant, reflect, mirror
      # frame = sx
      # k = np.array([[1,1,1],[1,1,0],[1,0,0]])
      # frame = ndimage.convolve(f, k, mode='constant', cval=0.0)
      # frame = ndimage.gaussian_laplace(f, sigma=1) # colored pixels
      # frame = ndimage.gaussian_laplace(f, sigma=0.1) # colored lsd
      # frame = ndimage.gaussian_laplace(f, sigma=0.01) # colored lsd
      
      # frame = ndimage.gaussian_laplace(f, sigma=0.001) # colored lsd

      # frame = cv.xphoto.oilPainting(f, 7, 1)
      dst_gray, dst_color = cv.pencilSketch(f, sigma_s=20, sigma_r=0.01, shade_factor=0.03) 
      frame = dst_gray
      # # f = np.resize(f, (135,240,3))
      # # f = np.resize(f, (55,20,3))
      # f = np.resize(f, (640,360,3))
      # # f.resize((240,135))
      # s = f.shape
      # for i in range(s[0]-1,0,-1):
      #   for j in range(s[1]-1,0,-1):
      #     # if f[i,j,0]-f[i-1,j-1,0] < 50:
      #     #   f[i-1,j-1,0] = f[i,j,0]
      #     # if f[i,j,1]-f[i-1,j-1,1] < 40:
      #     #   f[i-1,j-1,1] = f[i,j,1]
      #     if f[i,j,2]-f[i-1,j-1,2] < 90:
      #       f[i-1,j-1,2] = f[i,j,2]
      # f = np.resize(f, (1080,1920,3))
      
      # for i in range(s[0]-1,0,-1):
      #   for j in range(s[1]-1,0,-1):
      #     if f[i,j,0]-f[i-1,j-1,0] < 50:
      #       f[i-1,j-1,0] = f[i,j,0]
      #     if f[i,j,1]-f[i-1,j-1,1] < 40:
      #       f[i-1,j-1,1] = f[i,j,1]
      #     if f[i,j,2]-f[i-1,j-1,2] < 90:
      #       f[i-1,j-1,2] = f[i,j,2]
      # frame = f
      

      # frame = ndimage.gaussian_laplace(f, sigma=0.2) # colored

      # sy = ndimage.sobel(f, axis=1, mode='reflect')
      # sob = np.hypot(sx, sy)
      # frame = np.uint8(sob)
      # label_im, nb_labels = ndimage.label(f)
      # frame = label_im
      # sx = ndimage.sobel(f, axis=0, mode='constant')
      # sy = ndimage.sobel(f, axis=1, mode='constant')
      # sob = np.hypot(sx, sy)
      # frame = ndimage.rotate(f, 15, mode='constant') # slow
      # frame = sx - sy
      # frame = sy - sx
      # print(type(f))
      # print(len(f))
      # print(len(f[0]))
      # print(len(f[0][0]))
      # print(f[0][0][0])
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
      colors = list(self.colors) if self.colors else [(0, 255, 0)]
      if self.current_lyric_color:
        lyric_color = self.current_lyric_color
        colors.remove((lyric_color[2],lyric_color[1],lyric_color[0]))
      frame[dark_mask != 0] = random.choice(colors)
      frame[light_mask != 0] = random.choice(colors)
    return frame

  def change_current_image_filter(self, flter=None):
    if flter:
      self.current_image_filter = flter
    elif self.selected_filters:
      filters = list(self.selected_filters)
      filters.remove(self.current_image_filter)
      shuffle(filters)
      self.current_image_filter = filters[0]
    else:
      filters = list(self.image_filters)
      filters.remove(self.current_image_filter)
      self.current_image_filter = random.choice(filters)

  def change_current_video_filter(self, filter_name=None):
    if filter_name:
      self.current_video_filter = filter_name
    elif self.selected_filters:
      filters = list(self.selected_filters)
      filters.remove(self.current_video_filter)
      shuffle(filters)
      self.current_video_filter = filters[0]
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


  def prepare_main_video_loop(self):
    if self.output_name:
      self.log('progress', 'working on: {}'.format(self.output_name))
    # self.peak_frames = [120*i for i in range(100)]
    self.current_categories = self.main_categories.copy()
    self.current_peak_categories = self.peak_categories.copy()
    episodes_categories = self.parse_episodes_categories()
    all_categories = list(set(self.main_categories+self.peak_categories+episodes_categories))
    self.frame_queue.load_image_categories(all_categories)
    self.frame_queue.load_video_categories(all_categories)
    self.log('asset', 'all categories: {}'.format(all_categories))

    self.current_category = self.current_categories[0] if len(self.current_categories) else None
    self.peak_duration = 6
    self.image_source_rate = 6
    if self.live_video:
      self.total_video_frames = 30*60*60 # one hour long for clips
    self.episode_starts = [ e['start'](self.total_video_frames) if 'function' in str(type(e['start'])) else e['start'] for e in self.clips ]
    self.lyric_starts = [ l['start'] for l in self.lyrics ]
    self.lyric_ends = [ l['end'] for l in self.lyrics ]
    self.frame_count = 0
    self.peak_frame_count = 0
    self.peaks_enabled = False
    self.render_active = True
    self.log('structure', 'episode starts: {}'.format(self.episode_starts))
    self.log('structure', 'lyrics starts: {}'.format(self.lyric_starts))



  # OTHER LOOPS

  def generate_text_sprite(self, text, seconds=5, colors=[], background=(0,255,0)):
    # self.load_fonts(size=110, variation=0)
    self.load_fonts(text)
    self.background = background
    if colors:
      self.colors = colors
    else:
      self.colors = [
        (0,0,0),
        (255,255,255),
        (0,0,255),
        (255,255,0),
      ]
    currentDT = datetime.datetime.now()
    self.output_name = self.generate_output_name(text+'_sprite_'+currentDT.strftime("%y%m%d_%H%M%S"), 'text_sprites')
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    self.total_video_frames = seconds * self.fps
    self.filters_enabled = False
    self.video_enabled = False
    self.text_sprite = text
    self.main_video_loop(live_video=False)


  def generate_audio_roll(self, file_name, output_name=None, clips=[]):     
    currentDT = datetime.datetime.now()
    audio_file_path = self.get_audio_file_path(file_name)
    if not audio_file_path:
      print('no audio file')
      return
    self.total_video_frames, self.peak_frames = self.get_audio_metadata(audio_file_path)
    if output_name:
      self.output_name = self.generate_output_name(output_name+'_'+file_name[:-4]+'_'+currentDT.strftime("%y%m%d_%H%M%S"), 'audiovisual')
    else:
      self.output_name = self.generate_output_name(file_name[:-4]+'_'+currentDT.strftime("%y%m%d_%H%M%S"), 'audiovisual')
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    source_rate = self.fps * 13
    # if filters:
    #   self.filters_enabled = True
    # else:
    self.filters_enabled = False      
    self.video_enabled = True
    self.text_sprite = None
    self.background = None
    self.video_source_rate = source_rate
    self.image_source_rate = self.fps * 2 # source_rate
    self.frame_queue.update_source_rate(source_rate, image_source_rate=self.image_source_rate)
    self.main_video_loop(live_video=False)
    self.join_audio_video(self.output_name, audio_file_path)
    


  def generate_b_roll(self, seconds=5, source_rate=30, image_source_rate=None, filters=[]):     
    currentDT = datetime.datetime.now()
    self.output_name = self.generate_output_name('b_roll_'+currentDT.strftime("%y%m%d_%H%M%S"), 'b_roll')
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    self.total_video_frames = seconds * self.fps
    if filters:
      self.filters_enabled = True
    else:
      self.filters_enabled = False      
    self.video_enabled = True
    self.text_sprite = None
    self.background = None
    self.video_source_rate = source_rate
    self.image_source_rate = source_rate
    if image_source_rate:
      self.image_source_rate = image_source_rate
    self.frame_queue.update_source_rate(source_rate, image_source_rate=image_source_rate)
    self.main_video_loop(live_video=False)


  def set_video_dimensions(self, width, height):
    self.master_width = width
    self.master_height = height
    self.master_size = (self.master_width, self.master_height)


  def generate_instagram_video(self, text, clips, seconds=10, colors=[], background=(0,255,0)):
    # Square Photo  1:1 1080 x 1080px
    # Landscape Photo 1.91:1  1080 x 608px
    # Portrait Photo  4:5 1080 x 1350px
    # Instagram Stories 9:16  1080 x 1920px
    # IGTV Cover Photo  1:1.55  420 x 654px
    # Instagram Square Video  1:1 1080 x 1080px
    # Instagram Landscape Video 1.91:1  1080 x 608px
    # Instagram Portrait Video  4:5 1080 x 1350px
    # self.set_video_dimensions(1080, 1920) #stories, tiktok
    self.set_video_dimensions(1080, 1080) #post
    self.load_fonts(size=110, variation=0)
    # self.background = background
    if clips:
      self.clips = clips
    if colors:
      self.colors = colors
    else:
      self.colors = [
        (0,0,0),
        (255,255,255),
        (0,0,255),
        (255,255,0),
      ]
    self.output_name = self.generate_output_name('_'.join(text.split(' '))+'_ig', 'instagram')
    self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
    self.total_video_frames = seconds * self.fps
    self.video_enabled = True
    self.text_sprite = text
    self.video_source_rate = 25
    self.image_source_rate = 5
    self.frame_queue.update_source_rate(self.video_source_rate, image_source_rate=self.image_source_rate)
 
    self.main_video_loop(live_video=False)


  def multicam_video_loop(self):
    # PLAN AND IMPLEMENT
    # if i_f%5 == 0:
    #   frame2, current_frame = self.frame_queue.get_frame(image=True, category=self.current_category)
    #   shape2 = frame2.shape
    #   if shape2[0] > shape2[1]:
    #     # encontrar ratio de height
    #     ratio = shape2[0]/self.master_width
    #   else:
    #     # encontrar ratio de width
    #     ratio = shape2[1]/self.master_height
    #   frame2 = cv.resize(frame2, (int(self.master_width* ratio) , int(self.master_height*ratio)))
    #   shape2 = frame2.shape
    #   # to fit?
    #   frame2 = frame2[int(shape2[0]/2) - int(self.master_height/2):int(shape2[0]/2) + int(self.master_height/2), int(shape2[1]/2) - int(self.master_width/2):int(shape2[1]/2) + int(self.master_width/2)]
    #   final_frame2 = cv.resize(frame2, (self.master_width, self.master_height))

    # BLEND CAMS
    # final_frame = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)  
    # alpha = 0.3
    # # [blend_images]
    # beta = (1.0 - alpha)
    # blended = cv.addWeighted(final_frame, alpha, final_frame2, beta, 0.0)
    # # (final_frame*255).astype(np.uint8)
    # # final_frame  = cv.cvtColor(final_frame, cv.COLOR_HSV2BGR)
    # # final_frame = self.noisy('gauss',final_frame)
    # final_frame = np.uint8(blended)
    pass


  def batch_transform_videos(self, output_path, seconds=5, resize=False, zoom=1.35):
    for p in self.loaded_videos:
      file_name = p.split('/')[-1].split('.')[0]
      self.log('progress', 'working on: {}'.format(file_name))
      self.loaded_video = cv.VideoCapture(p)
      self.total_video_frames = int(self.loaded_video.get(cv.CAP_PROP_FRAME_COUNT))
      frame_count = 0
      written_frame_count = 0
      frame, current_frame = self.frame_queue.get_frame()
      segment_count = 0
      output_name = '{}_{}.mp4'.format(file_name, segment_count)
      segment_frames = seconds * self.fps
      if (self.total_video_frames/self.fps)/60 > 4:
        segment_frames = 30 * self.fps
      self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                       self.fourcc, 
                                       self.fps, 
                                       (frame.shape[1],frame.shape[0]), 
                                       self.color_video)
      while frame is not None:
        if resize and zoom:
          shape = frame.shape
          if shape[0] > shape[1]:
            ratio = self.master_width/shape[1]
          else:
            ratio = self.master_height/shape[0]
          (h, w) = frame.shape[:2]
          center = (w / 2, h / 2)
          # rotate the image by 180 degrees
          frame = cv.resize(frame, (self.master_width*3, 3*self.master_height), interpolation=cv.INTER_LINEAR)
          M = cv.getRotationMatrix2D(center, 0, ratio)
          upscale = cv.warpAffine(frame, M, (w, h))
          (r_h, r_w) = upscale.shape[:2]
          frame = cv.resize(upscale, (self.master_width, self.master_height), interpolation=cv.INTER_LINEAR)
          if f_c == 0:
            print(frame.shape)
        self.video_writer.write(frame) 
        
        if current_frame%segment_frames == 0 and current_frame+(segment_frames/2) < self.total_video_frames:
          self.destroy_video_writer()
          segment_count += 1
          output_name = '{}_{}.mp4'.format(file_name, segment_count)          
          self.video_writer = cv.VideoWriter(join_paths(output_path,output_name), 
                                            self.fourcc, 
                                            self.fps, 
                                            (frame.shape[1],frame.shape[0]), 
                                            self.color_video)
        
        frame, current_frame = self.frame_queue.get_frame()
      self.destroy_video_writer()  
      if self.loaded_video:
        self.loaded_video.release()



  def clip_podcast_video(self, input_path, clips=[]):
    # self.set_video_dimensions(1080, 1080) #post
    # self.load_fonts(size=110, variation=0)
    # self.background = background

    if clips:
      self.clips = clips
    # if colors:
    #   self.colors = colors
    # else:
    #   self.colors = [
    #     (0,0,0),
    #     (255,255,255),
    #     (0,0,255),
    #     (255,255,0),
    #   ]

    self.video_enabled = True
    self.filters_enabled = False

    for clip in clips:
      self.output_name = self.generate_output_name('_'.join(input_path.split(' '))+'_'+'_'.join(clip['options']['text'].split(' '))+'_clip', 'clips')
      self.video_writer = cv.VideoWriter(self.output_name, 
                                       self.fourcc, 
                                       self.fps, 
                                       self.master_size,
                                       self.color_video)
      self.total_video_frames = clip['duration'] * self.fps
      # self.text_sprite = text
      # self.video_source_rate = 25
      # self.image_source_rate = 5
      # self.frame_queue.update_source_rate(self.video_source_rate, image_source_rate=self.image_source_rate)
      frame_queue_config = dict(selected=input_path, start_time=0)

      self.frame_queue.update_config(frame_queue_config)
      self.main_video_loop(live_video=False)
      real_input_path = [ p for p in self.frame_queue.loaded_videos if input_path in p ][0]
      self.join_timed_audio_video(self.output_name, real_input_path)
    

  # MAIN LOOP

  def main_video_loop(self, live_video=True):
    self.live_video = live_video
    # cv.namedWindow('output', cv.WND_PROP_FULLSCREEN)
    # cv.setWindowProperty('output', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    self.prepare_main_video_loop()
    # cv.namedWindow("output", cv.WINDOW_NORMAL)

    while self.render_active:
      # end if last frame or esc 
      if not self.live_video and self.frame_count == self.total_video_frames:
        self.render_active = False
        break
      elif self.live_video:
        k = cv.waitKey(33)
        if k == 27:
          self.render_active = False
          break
        if k == 32: # ' '
          self.frame_queue.load_next_video()
        if k == 102: # 'f'
          self.filters_enabled = not self.filters_enabled
        if k == 103: # 'g'
          # self.filters_enabled = not self.filters_enabled
          if self.current_image_filter == 'invert':
            self.current_image_filter = 'sort'
          else:
            self.current_image_filter = 'invert'
        if k == 104: # 'g'
          # self.filters_enabled = not self.filters_enabled
          self.current_image_filter = 'half-sort'

          # cv.namedWindow("output", cv.WINDOW_FULLSCREEN)
      if self.live_video:
        # wait for next frame according to fps
        self.clock.tick(self.fps)

      self.frame_queue.prepare_frame(self.frame_count, video=self.video_enabled)
      self.update_episode_options(self.frame_count)
      self.react_to_peak(self.frame_count)
      self.update_current_lyrics(self.frame_count)

      frame, current_frame = self.frame_queue.get_frame(background=self.background, image=(not self.video_enabled), category=self.current_category)

      final_frame = self.resize_frame_for_output(frame)
      
      final_frame = self.apply_filters(final_frame, self.frame_count)

      final_frame = self.apply_texts(final_frame, self.frame_count)
      
      final_frame = final_frame[:self.master_height, :self.master_width]

      if self.live_video:
        cv.imshow('output', final_frame)
      else:
        self.video_writer.write(final_frame) 
        if self.frame_count%int(self.total_video_frames/5) == 0:
          self.log('progress', str(int(self.frame_count))+'/'+str(int(self.total_video_frames)))
      self.frame_count += 1

    self.frame_queue.destroy()
    if self.live_video:
      cv.destroyAllWindows()
    else:
      self.destroy_video_writer()

