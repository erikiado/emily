
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join as join_paths

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

class AudioAnalyzer:
  
  paths = []
  loaded_mp3s = []
  loaded_wavs = []
  all_files = []

  def __init__(self, file_paths=[]):
    self.paths += file_paths
    self.get_file_paths()

  def get_file_paths(self):
    for p in self.paths:  
      only_files = [join_paths(p, f) for f in listdir(p) if isfile(join_paths(p, f))]
      only_mp3s = list(filter(lambda f: '.mp3' in f.lower(), only_files))
      only_wavs = list(filter(lambda f: '.wav' in f.lower(), only_files))
      self.loaded_mp3s += only_mp3s
      self.loaded_wavs += only_wavs
      self.all_files = self.loaded_mp3s + self.loaded_wavs
      shuffle(self.all_files)
  
  def load_wav_as_signal(self, file_path):
    spf = wave.open(file_path, "r")
    signal = spf.readframes(-1)
    # signal = np.frombuffer(signal, "Int32")
    signal = np.frombuffer(signal, "Int16")
    return spf, signal
  

  def get_audio_metadata(self, audio_file_path, video_frames):
    print('analyzing:', audio_file_path)
    spf, signal = self.load_wav_as_signal(audio_file_path)
    fs = spf.getframerate() # 44000 frames in a second
    len_seconds = spf.getnframes() / spf.getframerate() # float
    timeX = np.linspace(0, len(signal) / fs, num=len(signal)) 
    peak_distance = 35000
    peaks, _ = find_peaks(signal, distance=peak_distance)
    peak_frames = [ int(ceil(p)) for p in ( peaks / (fs/video_frames) )]
    audio_duration = len_seconds
    return audio_duration, peak_frames
  

  def plot_audio_file_peaks(self, audio_file_path, video_frames):
    print('analyzing:', audio_file_path)
    spf, signal = self.load_wav_as_signal(audio_file_path)
    fs = spf.getframerate() # 44000 frames in a second
    len_seconds = spf.getnframes() / spf.getframerate() # float
    timeX = np.linspace(0, len(signal) / fs, num=len(signal)) 

    # print('signal:', signal)
    # print('length seconds:', len_seconds)
    # print('timeX:', timeX)
    # print('len timex:', len(timeX))
    # for x in (peaks/spf.getframerate()):
    #   print(x, len_seconds)

    # bands = array([0,3500,4000,5500,6000,fs/2.0]) / fs
    # desired = [0, 2, 0]
    # b = remez(513, bands, desired)
    # sig_filt = lfilter(b, 1, signal)
    # sig_filt /=  1.05 * max(abs(sig_filt)) # normalize
    # plt.plot(timeX, signal)
    # ax.title(str(loaded_wav))
    # ax.plot(timeX, sig_filt)
    # plt.plot(timeXpeaks, peaks)
    # peaks, _ = find_peaks(sig_filt, height=0, threshold=60000000)
    # ax.plot(peaks/spf.getframerate(), signal[peaks], "x")

    # plt.figure(1)
    # plt.title(loaded_wav)

    # # Draw X on peaks
    # ax = plt.subplot(2, 1, 2)
    # # ax.set_figheight(15)
    # # ax.set_figwidth(15)
    # ax.plot(timeX, signal)
    # ax.set_ylim(bottom=0.) # only positive waveform
    # ax.plot(peaks/spf.getframerate(), signal[peaks], ".")


    # signal = signal[:440000]
    # peak_distance = int(max(signal)/1000)
    peak_distance = 35000
    peaks, _ = find_peaks(signal, distance=peak_distance)

    for i, p in enumerate(peaks[:4]):
      print(i, 'p in audio', p)
    for i, p in enumerate((peaks/fs)[:4]):
      print(i, 'p in seconds', p)
    print()
    for i, p in enumerate(((peaks/(fs/30)))[:4]):
      print(i, 'p in 30fps video', int(ceil(p)))
    peak_frames = [ int(ceil(p)) for p in ( peaks / (fs/video_frames) )]
    print(peak_frames)

    
    peak_prominence = int(max(signal)) * 1.5
    peaks2, _ = find_peaks(signal, prominence=peak_prominence)      # BEST!
    peak_width = 200
    peaks3, _ = find_peaks(signal, width=peak_width)
    peak_thresh = int(max(signal)/40)
    peaks4, _ = find_peaks(signal, threshold=peak_thresh)     # Required vertical distance to its direct neighbouring samples, pretty useless
    print(len(peaks),'peaks')
    print(len(peaks2),'peaks2')
    print(len(peaks3),'peaks3')
    print(len(peaks4),'peaks4')
    plt.subplot(2, 2, 1)
    plt.plot(peaks, signal[peaks], "xr")
    # plt.plot(signal);
    plt.legend(['distance'])
    plt.subplot(2, 2, 2)
    plt.plot(peaks2, signal[peaks2], "ob")
    # plt.plot(signal);
    plt.legend(['prominence'])
    plt.subplot(2, 2, 3)
    plt.plot(peaks3, signal[peaks3], "vg")
    # plt.plot(signal);
    plt.legend(['width'])
    plt.subplot(2, 2, 4)
    plt.plot(peaks4, signal[peaks4], "xk")
    # plt.plot(signal);
    plt.legend(['threshold'])

    # Display full plot
    plt.show()



if __name__ == "__main__":
  anal = AudioAnalyzer(
    file_paths = [
      '/Users/erikiado/Code/emily/memories/',
    ])


  # anal.get_peaks_for_audiovisual()
  anal.plot_peaks_for_audiovisual('/Users/erikiado/Code/emily/memories/softy.wav', video_frames=30)

 




# # convert mp3, read wav
# mp3filename = 'XC124158.mp3'
# wname = mktemp('.wav')
# check_call(['avconv', '-i', mp3filename, wname])
# sig, fs, enc = wavread(wname)
# os.unlink(wname)



# subplot(211)
# specgram(sig, Fs=fs, NFFT=1024, noverlap=0)
# axis('tight'); axis(ymax=8000)
# title('Original')
# subplot(212)
# specgram(sig_filt, Fs=fs, NFFT=1024, noverlap=0)
# axis('tight'); axis(ymax=8000)
# title('Filtered')
# show()

# play(sig_filt, fs)










# image to sound

