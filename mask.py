#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# 一行做softmax
def mask_onlytime(m1,m2,m3,m4,imput):   # vocal, drum, .... dim(m1)=dim(m2)=... = time * amplitude
    all = np.c_[m1, m2, m3, m4]
    sf = tf.nn.softmax(all)
    mask1,mask2,mask3,mask4 = np.split(sf, 4, 1)
    m1 = mask1 * imput
    m2 = mask2 * imput
    m3 = mask3 * imput
    m4 = mask4 * imput
    return m1,m2,m3,m4


o1 = o2 = o3 = o4 = []

# 每4个数字做softmax
def mask(m1, m2, m3, m4, imput):  # vocal, drum, .... dim(m1)=dim(m2)=... = time * amplitude
    x, y = m1.shape
    all0 = np.c_[m1[0], m2[0], m3[0], m4[0]]
    sf0 = tf.nn.softmax(all0)
    ms10, ms20, ms30, ms40 = np.split(sf0, 4, 1)
    o1 = tf.transpose(ms10)
    o2 = tf.transpose(ms20)
    o3 = tf.transpose(ms30)
    o4 = tf.transpose(ms40)

    for time in range(1, x):
        al = np.c_[m1[time], m2[time], m3[time], m4[time]]
        sf = tf.nn.softmax(al)
        ms1, ms2, ms3, ms4 = np.split(sf, 4, 1)

        o1 = tf.concat([o1, tf.transpose(ms1)], 0)
        o2 = tf.concat([o2, tf.transpose(ms2)], 0)
        o3 = tf.concat([o3, tf.transpose(ms3)], 0)
        o4 = tf.concat([o4, tf.transpose(ms4)], 0)

    m1 = o1 * imput
    m2 = o2 * imput
    m3 = o3 * imput
    m4 = o4 * imput
    return m1, m2, m3, m4

