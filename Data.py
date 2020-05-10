from IPython.display import Audio, display
import urllib.request
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
from urllib.request import urlopen
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.transforms import BlendedGenericTransform
import scikit_posthocs as sp
from math import ceil
from scipy.signal import stft, istft
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from google.colab import drive
# drive.mount('/content/drive')

import musdb

def chop(spectrogram, wanted_time, ratio_overlap):
  # 条件： wanted_time * ratio_overlap 是整数
  (I, F, T) = spectrogram.shape
  K = int(ceil((T / wanted_time - ratio_overlap)/(1 - ratio_overlap)))  # ceil取上,要分成多少块
  expanded_T = int(wanted_time + wanted_time * (1 - ratio_overlap) * (K - 1)) # 扩展后的time维度
  newSpectrogram = np.zeros((I, F, expanded_T))   # 右：多一些0，至能被girdsize整除
  newSpectrogram[:, :, :T] = spectrogram
  slices = np.empty((0, I, F, wanted_time), float)
  for k in range(0, K):   # 列
    s = newSpectrogram[:, :, round(k * wanted_time * (1-ratio_overlap)):round(k * wanted_time * (1-ratio_overlap) + wanted_time)] # 切为很多小块
    s = np.expand_dims(s, axis=0)
    slices = np.append(slices, s, axis=0)
  slices = np.transpose(slices, (0, 1, 3, 2)) # 想要[batch, channel, time, freaquency] 
  return slices

def dataset(nbmus, time=256, ratio_overlap=0.5):
  eps = np.finfo(np.float).eps  # small epsilon to avoid dividing by zero

  X = np.empty((0, 2, time, 2049), float)
  M = {'vocals': np.empty((0, 2, time, 2049), float),
      'drums': np.empty((0, 2, time, 2049), float),
      'bass': np.empty((0, 2, time, 2049), float),
      'other': np.empty((0, 2, time, 2049), float)} # [batch, channel, time, freaquency] 

  for track in nbmus:
    # print(track.name)
    x = np.abs(stft(track.audio.T, nperseg=4096, noverlap=3072)[-1])   # shape: (nb_channels=channel, nb_features=freq, nb_frames=time)
    xs = chop(x,time,ratio_overlap)
    X = np.append(X, xs, axis=0)

    P = {} # sources spectrograms
    # compute model as the sum of spectrograms 分母
    model = eps
    for name, source in track.sources.items():  # 遍历所有声部，求mask中的分母
      # compute spectrogram of target source:
      P[name] = np.abs(stft(source.audio.T, nperseg=4096, noverlap=3072)[-1])
      model += P[name]

    for name, source in track.sources.items(): # 遍历所有声部，用mask分离出各个声部
      # compute soft mask as the ratio between source spectrogram and total
      mask = P[name] / model
      masks = chop(mask,time,ratio_overlap)
      M[name] = np.append(M[name], masks, axis=0)

  return X, M

def ichop(X_origin, M, time_scale=256, ratio_overlap=0.5): # 输入只一首歌
  channel, frequency, time = X_origin.shape
  newM = {'vocals': np.empty((channel, frequency, time+time_scale), float),
      'drums': np.empty((channel, frequency, time+time_scale), float),
      'bass': np.empty((channel, frequency, time+time_scale), float),
      'other': np.empty((channel, frequency, time+time_scale), float)} # [channel, frequency, time] 
  for name, source in M.items():
    for i in range(source.shape[0]): # 遍历batch
      toInput = np.transpose(source[i,:,:,:], (0, 2, 1))  # (2, 2049, 240)
      if i == 0:
        newM[name][:,:, i:i + time_scale] = toInput
      else:
        
        newM[name][:,:, int(i * time_scale * (1 - ratio_overlap)):int((i * time_scale * (1 - ratio_overlap) + time_scale * ratio_overlap))] = np.true_divide(
              np.add(
              newM[name][:,:, int(i * time_scale * (1 - ratio_overlap)):int(i * time_scale * (1 - ratio_overlap) + time_scale * ratio_overlap)],toInput[:,:, :int(time_scale * ratio_overlap)]), 
              2)
        newM[name][:,:, int(i * time_scale * (1 - ratio_overlap) + time_scale * ratio_overlap):int(
              i * time_scale * (1 - ratio_overlap) + time_scale)] = toInput[:,:,-int(time_scale * (1 - ratio_overlap)):]
    newM[name] = newM[name][:channel, :frequency, :time]
  return newM

def estimateSpectro(X_origin, newM):
  
  # small epsilon to avoid dividing by zero
  eps = np.finfo(np.float).eps
  # compute model as the sum of spectrograms
  model = eps

  for name, source in newM.items():  # 遍历所有声部，求mask中的分母
    model += newM[name]


  # now performs separation
  estimates = {}
  for name, source in newM.items(): # 遍历所有声部，用mask分离出各个声部
    # compute soft mask as the ratio between source spectrogram and total
    Mask = newM[name] / model

    # multiply the mix by the mask
    Yj = Mask * X_origin

    # invert to time domain
    target_estimate = istft(Yj, nperseg=4096, noverlap=3072)[1].T

    # set this as the source estimate
    estimates[name] = target_estimate

  return estimates

