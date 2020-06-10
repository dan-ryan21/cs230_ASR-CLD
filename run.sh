##############################################################
#  Dan Ryan
#
#  Stanford University - CS230 Deep Learning
#
#  06/05/2020
#
#  dryan2@stanford.edu
#  dan_ryan21@hotmail.com
#  
##############################################################


###  Build training audio files and labeled outputs
python asr-cld/generate_dataset.py

###  Create numpy objects for train/dev input/output files
python asr-cld/generate_train_dev_sets.py

###  Train the model (inputs = batch size, epochs)
python asr-cld/train_model.py 200 10

###  Evaluate the model
python asr-cld/evaluate_model.py