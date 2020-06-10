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
#  Creates a list of all audio files, and their corresponding text, from the LibriSpeech dataset that
#  don't contain any words greater than the specified length.  Takes as input the root directory of
#  the dataset and the maximum word length.
#
#####################################################################################################

import os
import sys

# Read input arguments
rootDirectory = sys.argv[1]
maxWordLength = int(sys.argv[2])

# Build paths for output files
audioFilesPath = rootDirectory + "/" + "input_files.txt"
audioTxtPath = rootDirectory + "/" + "input_labels.txt"

# Remove output files if they have already been created
if os.path.exists(audioFilesPath):
    os.remove(audioFilesPath)

if os.path.exists(audioTxtPath):
    os.remove(audioTxtPath)

# Initialize lists
count = 0
audioFiles = []
audioTxt = []

# Look for all samples with ONLY words less than max length
for baseFolder in os.listdir(rootDirectory):
    for folder in os.listdir(rootDirectory + "/" + baseFolder):
        for file in os.listdir(rootDirectory + "/" + baseFolder + "/" + folder):
            if file.endswith('.txt'):
                filepath = rootDirectory + "/" + baseFolder + "/" + folder + "/" + file
                f = open(filepath, 'r')
                lines = f.readlines()
                for line in lines:
                    count += 1
                    words = line.split()
                    audioFile = words.pop(0) + '.wav'
                    audioFiles.append(audioFile)
                    separator = ' '
                    audioTxt.append(separator.join(words))
                    for word in words:
                        if len(word) > maxWordLength:
                            count -= 1
                            audioFiles.pop()
                            audioTxt.pop()
                            break

# Write output files
with open(audioFilesPath, 'w') as filehandle:
    for af in audioFiles:
        filehandle.write('%s\n' % af)

with open(audioTxtPath, 'w') as filehandle:
    for l in audioTxt:
        filehandle.write('%s\n' % l)

# print(count)

