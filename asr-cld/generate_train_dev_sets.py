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
#  Builds the training and dev datasets used for the model.  Reads all of the previously generated
#  audio files, takes their spectrograms, and appends them together in one set.  Reads all of the
#  previously generated labeled files and appends them together in one set.  Saves these master sets
#  as npy files so that they can be easily loaded when training and evaluating the model.
#
#  Training data was created by randomly overlaying positive/negative search words over a random
#  background signal.  Dev data was recorded on a Logitech webcam microphone and labelled manually.
#
#  NOTE:  This process was modeled from the Coursera Trigger Word Detection programming assignment.
#  It has been modified to accomodate multiple labels.
#
#####################################################################################################

import os
import numpy as np
from audio_processing import *
from asr_cld_constants import *
from keras.utils import to_categorical

# Location of data files
trainDirectory = 'data/speech_commands_v0.01_edited/train'
trainAudioDirectory = trainDirectory + '/audio'
trainLabelDirectory = trainDirectory + '/labels'
trainXfile = trainDirectory + '/X.npy'
trainYfile = trainDirectory + '/Y.npy'
devDirectory = 'data/speech_commands_v0.01_edited/dev'
devAudioDirectory = devDirectory + '/audio'
devLabelDirectory = devDirectory + '/labels'
devXfile = devDirectory + '/X.npy'
devYfile = devDirectory + '/Y.npy'

# Load Constants
Tx = getTx()
Ty = getTy()
n_freq = getNfreq()
m_dev = getDevSetSize()
m_train = getTrainSetSize()
nc = getNumOfClasses()

# Initialize X-dev
X = np.empty((m_dev, Tx, n_freq))
i = 0

# Insert all dev samples in X
for wavFile in os.listdir(devAudioDirectory):
    wavPath = devAudioDirectory + '/' + wavFile
    x = graph_spectrogram(wavPath)
    x = np.transpose(x)
    X[i, :, :] = x
    i += 1

# Save X-dev
np.save(devXfile, X)

# Initialize Y-dev
Y = np.empty((m_dev, nc, Ty))
i = 0

# Insert all one-hot encoded dev labels in Y
for labelFile in os.listdir(devLabelDirectory):
    labelPath = devLabelDirectory + '/' + labelFile
    y = np.loadtxt(labelPath, delimiter=',')
    y_onehot = to_categorical(y, num_classes=nc)
    Y[i, :, :] = y_onehot
    i += 1

# Save Y-dev
np.save(devYfile, Y)

# Initialize X-train
X = np.empty((m_train, Tx, n_freq))
i = 0

# Insert all train samples in X
for wavFile in os.listdir(trainAudioDirectory):
    wavPath = trainAudioDirectory + '/' + wavFile
    x = graph_spectrogram(wavPath)
    x = np.transpose(x)
    X[i, :, :] = x
    i += 1

# Save X-train
np.save(trainXfile, X)

# Initialize Y-train
Y = np.empty((m_train, nc, Ty))
i = 0

# Insert all one-hot encoded train labels in Y
for labelFile in os.listdir(trainLabelDirectory):
    labelPath = trainLabelDirectory + '/' + labelFile
    y = np.loadtxt(labelPath, delimiter=',')
    y_onehot = to_categorical(y, num_classes=nc)
    Y[i, :, :] = y_onehot
    i += 1

# Save Y-train
np.save(trainYfile, Y)

