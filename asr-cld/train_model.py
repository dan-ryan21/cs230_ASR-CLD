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
#  Executes the training of the RNN model
#
#  NOTE:  Much of the code contained here is taken from the Coursera Trigger Word Detection
#  programming assignment.  It has been modified to accomodate multiple labels.
#
#####################################################################################################

from model import *
from asr_cld_constants import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import sys

# Read input arguments
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])

Tx = getTx()
n_freq = getNfreq()

model = getModel(input_shape=(Tx, n_freq))

# Display the model summary
model.summary()

# Compile the model, Adam optimizer
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# Get training data
X = np.load('data/speech_commands_v0.01_edited/train/X.npy')
Y = np.load('data/speech_commands_v0.01_edited/train/Y.npy')

# Setup checkpointing
model_directory = 'models/'
model_path = model_directory + 'tr_model.h5'

if not os.path.exists(model_directory):
    os.mkdir(model_directory)

checkpoint = ModelCheckpoint(filepath=model_path, monitor='acc', mode='max', save_best_only=True)

# Train the model
model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
