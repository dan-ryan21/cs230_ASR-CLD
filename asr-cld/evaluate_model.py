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
#  Tests the model on the dev set
#
#  NOTE:  Much of the code contained here is taken from the Coursera Trigger Word Detection
#  programming assignment.  It has been modified to accomodate multiple labels.
#
#####################################################################################################

from model import *
from asr_cld_constants import *
import numpy as np
from keras.optimizers import Adam

Tx = getTx()
n_freq = getNfreq()

# Load the saved model
model = getModel(input_shape=(Tx, n_freq))
model.load_weights('models/tr_model.h5')

# Compile the model, Adam optimizer
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# Get testing data
X_dev = np.load('data/speech_commands_v0.01_edited/dev/X.npy')
Y_dev = np.load('data/speech_commands_v0.01_edited/dev/Y.npy')

# Evaluate model on dev set
loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)

