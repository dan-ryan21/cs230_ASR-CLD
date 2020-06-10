# cs230_ASR-for-Childhood-Language-Development

#### This project is intended to run on an AWS Deep Learning AMI (Ubuntu 16.04) instance, preferably with at least one GPU (p2.xlarge recommended).  Before running, you should activate the tensorflow_p36 environment with the following statement:

* source activate tensorflow_p36

#### Additionally, you may need to manually install the following packages:

* pydub ------- pip install pydub
* ffmpeg ------ sudo apt-get install ffmpeg 

#### To begin, clone this repository and move to the root directory before executing any of the following commands

#### The model can be executed end-to-end through a script by executing the command below.  This script will build the train/dev datasets, train the model with default parameters (selected such that the model should train in ~ 10 mins), and then evaluate the model on the dev/test datasets.  Even though the model will be trained with default parameters, the model will be evaluated with our best model to-date.  To select a different model, see the evaluate model stage below.

* ./run.sh

#### Additionally, each stage in the pipeline can be executed individually.  The command below builds the dataset used for the training examples.  This involves overlaying randomly selected positive/negative words on a randomly selected background signal.

* python asr-cld/generate_dataset.py

#### The command below will then assemble the training and dev sets into numpy objects that can be quickly loaded during the training/evaluation stages

* python asr-cld/generate_train_dev_sets.py

#### The command below will then train the model with the specified batch size and number of epochs.  The model parameters will be saved to the /models directory with the default tr_model.hr filename.

* python asr-cld/train_model.py $batch_size $num_epochs

#### Finally, the model can be evaluated with the following command.  The model will be evaluated against the dev training set, with accuracy reported.  Then, predictions will be made against the test set and the predictions will be saved to files.  The name of the model to use can be passed as an argument.  If no argument is passed, the last training model will be used.  All models should be located in the /models directory.

* python asr-cld/evaluate_model.py $model_name
