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

Tx = getTx()
n_freq = getNfreq()

# Load the saved model
model = getModel(input_shape=(Tx, n_freq))
model.load_weights('models/tr_model.h5')
