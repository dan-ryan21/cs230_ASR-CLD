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
import numpy as np
import os

Tx = getTx()
n_freq = getNfreq()

model = getModel(input_shape=(Tx, n_freq))

# Display the model summary
model.summary()

# Use an Adam optimizer
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# Get training data
X = np.load('data/speech_commands_v0.01_edited/train/X.npy')
Y = np.load('data/speech_commands_v0.01_edited/train/Y.npy')

# Train the model
model.fit(X, Y, batch_size=50, epochs=100)

# Save the model
if not os.path.exists('models/'):
    os.mkdir('models/')

model.save_weights('models/tr_model.h5')
