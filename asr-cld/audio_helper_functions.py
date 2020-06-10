#####################################################################################################
#  Dan Ryan
#
#  Stanford University - CS230 Deep Learning
#
#  06/05/2020
#
#  dryan2@stanford.edu
#  dan_ryan21@hotmail.com
#
#  Contains helper functions used to generate the audio samples in the dataset
#
#  NOTE:  Much of the code contained here is taken from the Coursera Trigger Word Detection
#  programming assignment.  It has been modified to accomodate multiple labels.
#
#####################################################################################################

import numpy as np
from asr_cld_constants import *


# Gets a random time segment of duration segment_ms
def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=getAudioLength() - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


# Checks if the time of a segment overlaps with the times of existing segments.
def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time

    overlap = False

    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


# Insert a new audio segment over the background noise at a random time step
def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)

    segment_time = get_random_time_segment(segment_ms)

    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)

    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


# Update the label vector y
def insert_label(y, label, segment_end_ms):
    Ty = getTy()
    segment_end_y = int(segment_end_ms * Ty / getAudioLength())

    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = label

    return y


# Adjusts the gain of an audio sample
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Creates a training example with a given background, positives, and negatives.
def create_training_example(backgrounds, positives, labels, negatives):
    Ty = getTy()
    y = np.zeros((1, Ty))

    previous_segments = []

    background_index = np.random.randint(len(backgrounds))
    background = backgrounds[background_index]

    background = background - 40

    number_of_positives = np.random.randint(0, getNumOfPositives())
    random_indices = np.random.randint(len(positives), size=number_of_positives)
    random_positives = [positives[i]['data'] for i in random_indices]
    random_labels = [labels.index(positives[i]['label']) + 1 for i in random_indices]

    for (random_positive, random_label) in zip(random_positives, random_labels):
        background, segment_time = insert_audio_clip(background, random_positive, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_label(y, random_label, segment_end)

    number_of_negatives = np.random.randint(0, getNumOfNegatives())
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i]['data'] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    background = match_target_amplitude(background, -20.0)

    return background, y
