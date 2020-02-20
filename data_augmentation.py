#!/usr/bin/env python
# -*- coding:utf-8 -*-
from specAugment import spec_augment_tensorflow
# Time Shift Augmentation
# spectrogram = time_shift_spectrogram(spectrogram)
def time_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)


# Pitch Shift Augmentation
# spectrogram = pitch_shift_spectrogram(spectrogram)
def pitch_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)

# It modifies the spectrogram by warping it in the time direction, masking
# blocks of consecutive frequency channels, and masking blocks of utterances in time.
# https://arxiv.org/pdf/1904.08779.pdf
# pip3 install SpecAugment
def SpecAugment(spectrogram):
    spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=spectrogram)
    return spectrogram
