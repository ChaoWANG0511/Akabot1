#!/usr/bin/env python
# -*- coding:utf-8 -*-

import librosa

filePath= 'Akabot\DSD100subset\Mixtures\Dev\055 - Angels In Amplifiers - I'm Alright\mixture.wav '

audio, sampleRate = librosa.load(filePath)

def chop(matrix, scale):
    slices = []
    for time in range(0, matrix.shape[1] // scale):   # 列
        for freq in range(0, matrix.shape[0] // scale):  # 行
            s = matrix[freq * scale : (freq + 1) * scale,
                       time * scale : (time + 1) * scale]
            slices.append(s)  # 切为很多小块，每块scale*scale，存到slices列表， 使得模型的输入都一样大
    return slices
 slices = chop()
