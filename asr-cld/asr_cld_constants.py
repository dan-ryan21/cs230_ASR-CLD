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
#  Getter methods for project constants
#
#####################################################################################################

# Length of audio samples in ms
def getAudioLength():
    return 10000


# Timesteps in spectrogram
def getTx():
    return 5511


# Frequencies in spectrogram
def getNfreq():
    return 101


# Timesteps in model output
def getTy():
    return 1375


# The number os samples in the train set
def getTrainSetSize():
    return 2000


# The number os samples in the train set
def getDevSetSize():
    return 26


# The number os samples in the test set
def getTestSetSize():
    return 5


# The maximum number of positive clips to insert
def getNumOfPositives():
    return 3


# The maximum number of negative clips to insert
def getNumOfNegatives():
    return 3


# The number of output classes
def getNumOfClasses():
    return 7
