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

##############################################################
#  Default File Locations
##############################################################

data_local=data
data_url=www.openslr.org/resources/12
train_set=dev-clean
#test_set=test-clean

##############################################################
#  Create Local Data Directory
##############################################################

if [ ! -e $data_local ]; then
    mkdir $data_local
fi

##############################################################
#  Download Data
##############################################################

. scripts/download.sh ${data_url} ${train_set} ${data_local}
#. scripts/download.sh ${data_url} ${test_set} ${data_local}

##############################################################
#  Decompress Data
##############################################################

. scripts/decompress.sh ${train_set} ${data_local}
#. scripts/decompress.sh ${test_set} ${data_local}

##############################################################
#  Convert .flac files to .wav
##############################################################

data_folder=$data_local/LibriSpeech/$train_set

. scripts/convert.sh ${data_folder}

##############################################################
#  Preprocess Data
##############################################################

max_word_length=200

python asr-cld/preprocess_files.py $data_folder $max_word_length
#python asr-cld/preprocess_files.py $data_local/LibriSpeech/$test_set 200
