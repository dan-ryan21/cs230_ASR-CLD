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
#  Builds the dataset used for training the model.  Each example consists of a random selection of
#  background noise, 10s in length.  Then, a random number of positive words are inserted into the
#  audio clip.  At the same time the labels are inserted in the output file.  Finally, a random
#  number of negative words are inserted into the audio clip.
#
#  NOTE:  Much of the code contained here is taken from the Coursera Trigger Word Detection
#  programming assignment.  It has been modified to accomodate multiple labels.
#
#####################################################################################################

import os
from pydub import AudioSegment
from audio_helper_functions import *
from audio_processing import *

# Location of audio files
backgroundDirectory = 'data/speech_commands_v0.01_edited/background'
positiveDirectory = 'data/speech_commands_v0.01_edited/positive'
negativeDirectory = 'data/speech_commands_v0.01_edited/negative'

# Create lists for audio samples
backgrounds = []
positives = []
negatives = []
labels = []

print("\n*** Loading Audio Clips")

# Build List of background samples
for backgroundFile in os.listdir(backgroundDirectory):
    path = backgroundDirectory + '/' + backgroundFile
    background = AudioSegment.from_wav(path)
    backgrounds.append(background)

# Build List of positive samples
for positiveLabel in os.listdir(positiveDirectory):
    if positiveLabel not in labels:
        labels.append(positiveLabel)
    labelPath = positiveDirectory + '/' + positiveLabel
    for positiveFile in os.listdir(labelPath):
        positivePath = labelPath + '/' + positiveFile
        positive = AudioSegment.from_wav(positivePath)
        dictEntry = {'label': positiveLabel, 'data': positive}
        positives.append(dictEntry)

# Build List of negative samples
for negativeLabel in os.listdir(negativeDirectory):
    labelPath = negativeDirectory + '/' + negativeLabel
    for negativeFile in os.listdir(labelPath):
        negativePath = labelPath + '/' + negativeFile
        negative = AudioSegment.from_wav(negativePath)
        dictEntry = {'label': negativeLabel, 'data': negative}
        negatives.append(dictEntry)

print("\n*** Generating Dataset\r"),

num_of_samples = getTrainSetSize()

if os.path.exists("data/speech_commands_v0.01_edited/train/"):
    os.system("rm -rf data/speech_commands_v0.01_edited/train/")

os.mkdir("data/speech_commands_v0.01_edited/train/")
os.mkdir("data/speech_commands_v0.01_edited/train/audio/")
os.mkdir("data/speech_commands_v0.01_edited/train/labels/")

for i in range(num_of_samples):
    data, y = create_training_example(backgrounds, positives, labels, negatives)

    file_handle = data.export("data/speech_commands_v0.01_edited/train/audio/train" + str(i) + ".wav", format="wav")
    np.savetxt("data/speech_commands_v0.01_edited/train/labels/label" + str(i) + ".csv", y, delimiter=",")

    if i % 200 == 0:
        print("*** Generating Dataset = " + str(i*100/num_of_samples) + "% Complete\r"),

    file_handle.close()

print("*** Generating Dataset = 100% Complete\n")
