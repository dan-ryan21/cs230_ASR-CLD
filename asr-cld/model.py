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
#  Definition of the model used to detect keywords.  The model takes as input a spectrogram and
#  outputs a time series of labels corresponding to the words, if any, detected.  The labels for
#  this implementation are as follows:
#
#  0 - No word or unrecognized word detected
#  1 - 'bird'
#  2 - 'cat'
#  3 - 'dog'
#  4 - 'one'
#  5 - 'two'
#  6 - 'three'
#
#  NOTE:  Much of the code contained here is taken from the Coursera Trigger Word Detection
#  programming assignment.  It has been modified to accomodate multiple labels.
#
#####################################################################################################

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape


#  Build the Keras Model
def getModel(input_shape):
    X_input = Input(shape=input_shape)

    # CONV layer
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(rate=0.8)(X)

    # GRU Layer
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)

    # GRU Layer
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.8)(X)

    # Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation="softmax"))(X)

    model = Model(inputs=X_input, outputs=X)

    return model
