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
from audio_processing import *
import numpy as np
from keras.optimizers import Adam
import os

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
print("*** Dev set accuracy = ", acc)

# Predict on test set
testAudioDirectory = 'data/speech_commands_v0.01_edited/test/audio'
testPredictDirectory = 'data/speech_commands_v0.01_edited/test/predict'

if not os.path.exists(testPredictDirectory):
    os.mkdir(testPredictDirectory)

X = np.empty((1, Tx, n_freq))
i = 0

for wavFile in os.listdir(testAudioDirectory):
    wavPath = testAudioDirectory + '/' + wavFile
    x = graph_spectrogram(wavPath)
    x = np.transpose(x)
    X[0, :, :] = x
    y = model.predict(X)
    prediction = np.argmax(y[0, :, :], axis=-2)
    np.savetxt(testPredictDirectory + "/predict" + str(i) + ".csv", prediction, delimiter=",")
    i += 1

print("\n*** Predictions on test set saved at data/speech_commands_v0.01_edited/test/predict")
