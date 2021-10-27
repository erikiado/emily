
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import struct
from os import listdir
from os.path import isfile, join as join_paths
from pathlib import Path

from random import shuffle

from scipy.signal import find_peaks

import wave
import sys


# import soundscrape
# soundscrape.process_soundcloud('https://soundcloud.com/dis_fig/shalt-x-supremes-unconfined-dis-fig-bootie-1')


# from scikits.audiolab import wavread, play
from scipy.signal import remez, lfilter
from pylab import *

# audio analyzer class
# get current volume
# conexion con interfaz

class AudioEditor:
  
  paths = []
  loaded_mp3s = []
  loaded_wavs = []
  all_files = []
  wav_values = None

  def __init__(self, file_paths=[], output_path=None):
    self.paths += file_paths
    self.get_file_paths()
    self.output_path = output_path if output_path else os.path.abspath(os.getcwd())
    self.wav_values = dict()


  def get_file_paths(self):
    for p in self.paths:  
      only_files = [join_paths(p, f) for f in listdir(p) if isfile(join_paths(p, f))]
      only_mp3s = list(filter(lambda f: '.mp3' in f.lower(), only_files))
      only_wavs = list(filter(lambda f: '.wav' in f.lower(), only_files))
      self.loaded_mp3s += only_mp3s
      self.loaded_wavs += only_wavs
      self.all_files = self.loaded_mp3s + self.loaded_wavs
      shuffle(self.loaded_mp3s)
      shuffle(self.loaded_wavs)
      shuffle(self.all_files)
  

  def generate_output_name(self, name, directory=None, dated=False):
    if dated:
      currentDT = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
      name += '_' + currentDT
    if directory:
      Path(join_paths(self.output_path, directory)).mkdir(parents=True, exist_ok=True)
    else:
      Path(self.output_path).mkdir(parents=True, exist_ok=True)
    # name += '_'+str(self.master_width)+'x'+str(self.master_height)+'.wav' 
    name += '.wav' 
      # only supporting mp4; update VideoWriter
    return join_paths(self.output_path, directory, name)

  def read_wav_values(self, filename):
    wave_file = wave.open(filename, 'r')
    nframes = wave_file.getnframes()
    nchannels = wave_file.getnchannels()
    sampling_frequency = wave_file.getframerate()
    T = nframes / float(sampling_frequency)
    read_frames = wave_file.readframes(nframes)
    wave_file.close()
    data = struct.unpack("%dh" %  nchannels*nframes, read_frames)
    return T, data, nframes, nchannels, sampling_frequency

  def store_wav_values(self, filename):
    if filename not in self.wav_values:
      T, data, nframes, nchannels, sampling_frequency = self.read_wav_values(filename)
      self.wav_values[filename] = dict(T=T,
                                       data=data,
                                       nframes=nframes,
                                       nchannels=nchannels,
                                       sampling_frequency=sampling_frequency)


  def concat_audios(self):
    if len(self.loaded_wavs) > 1:
      self.store_wav_values(self.loaded_wavs[0])
      self.store_wav_values(self.loaded_wavs[1])
      T2, data2, nframes2, nchannels2, sampling_frequency2 = self.read_wav_values(self.loaded_wavs[1])

      data_per_channel = [data[offset::nchannels] for offset in range(nchannels)]
      data2_per_channel = [data2[offset::nchannels2] for offset in range(nchannels2)]

      print('T', T)
      print('data', len(data))
      print('nf', nframes)
      print('nc', nchannels)
      print('sf', sampling_frequency)
      print('data_per_channel', len(data_per_channel[0]), len(data_per_channel[1]))
      total_frames = None
      if nframes > nframes2:
        total_frames = nframes
      else:
        total_frames = nframes2

      # total_frames = int(total_frames/4)
      final_data = [[],[]]

      output_name = self.generate_output_name('test', directory='mixed_wavs', dated=True)
      spf3 = wave.open(output_name, 'w')
      spf3.setnchannels(2)
      spf3.setsampwidth(2)
      spf3.setframerate(44100)

      for i in range(total_frames):
        try:
          a = data_per_channel[0][i]
          b = data2_per_channel[1][i]
          final_data[0].append(a)
          final_data[1].append(b)

          # data_per_channel[1]


          # data = []
          # segment_length = 10000 * 2
          # for i in range(1000000):
          #   d = None
            # if i%segment_length == int(segment_length/2):
            #   data.append(signal1[i])
            #   d = signal1[i]
            # else:
            #   data.append(signal2[i])
            #   d = signal2[i]
          segment_length =800
          if i%segment_length<segment_length/3:
            c = a
          elif i%segment_length<2*(segment_length/3):
            c = b
          else:
            c = int((a+b)/2)
          
          a = c
          b = c

          frame = struct.pack('<h', a)
          spf3.writeframes(frame)
          frame = struct.pack('<h', b)
          spf3.writeframes(frame)
        except Exception as e:
          print(e)
          print(i)
          break
      spf3.close()

  def concat_aud2(self):
    print(len(self.loaded_wavs))
    if len(self.loaded_wavs) > 1:
      T, data, nframes, nchannels, sampling_frequency, lc, rc = self.read_values(self.loaded_wavs[0])
      print('T', T)
      print('data', len(data))
      print('nf', nframes)
      print('nc', nchannels)
      print('sf', sampling_frequency)
      print('lc', lc)
      print('rc', rc)
      spf1 = wave.open(self.loaded_wavs[0], "r")
      signal1 = spf1.readframes(-1)
      # signal1 = spf1.readframes(100100)
      signal1 = np.frombuffer(signal1, "Int16")

      spf2 = wave.open(self.loaded_wavs[1], "r")
      signal2 = spf2.readframes(-1)
      # signal2 = spf2.readframes(100100)
      signal2 = np.frombuffer(signal2, "Int16")

      print(self.loaded_wavs[0])
      print(spf1)
      print(self.loaded_wavs[1])
      print(spf2)
      print('samp_width1:',spf1.getsampwidth())
      print('samp_channels1:', spf1.getnchannels())
      print('samp_rate1:', spf1.getframerate())
      print('n_frames1:', spf1.getnframes())
      print('samp_width2:',spf2.getsampwidth())
      print('samp_channels2:', spf2.getnchannels())
      print('samp_rate2:', spf2.getframerate())
      print('n_frames2:', spf2.getnframes())
      output_name = self.generate_output_name('test', directory='mixed_wavs')
      spf3 = wave.open(output_name, 'w')
      spf3.setnchannels(2)
      spf3.setsampwidth(2)
      spf3.setframerate(44100)

      data = []
      segment_length = 10000 * 2
      for i in range(1000000):
        d = None
        if i%segment_length == int(segment_length/2):
          data.append(signal1[i])
          d = signal1[i]
        else:
          data.append(signal2[i])
          d = signal2[i]
        frame = struct.pack('<h', d)
        spf3.writeframes(frame)
      spf3.close()



if __name__ == "__main__":
  anal = AudioEditor(
    file_paths = [
      '/Users/erikiado/Code/emily/memories/',
    ])


  # anal.get_peaks_for_audiovisual()
  anal.plot_peaks_for_audiovisual('/Users/erikiado/Code/emily/memories/softy.wav', video_frames=30)

 
