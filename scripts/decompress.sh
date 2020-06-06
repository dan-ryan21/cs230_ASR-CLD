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
#  Check Inputs
##############################################################

if [ $# -ne 2 ]; then
    echo "Error.  Expected 2 inputs to decompress.sh"
    exit 1
fi

##############################################################
#  Assign Inputs
##############################################################

data_set=$1
data_local=$2

##############################################################
#  Decompress Data
##############################################################

if ! tar -C $data_local -xvzf $data_local/$data_set; then
    echo "Error. Un-tarring archive $data_set failed"
    exit 1
fi

echo "Successfully un-tarred archive $data_set"
