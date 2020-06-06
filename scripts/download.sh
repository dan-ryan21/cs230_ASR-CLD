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

if [ $# -ne 3 ]; then
    echo "Error.  Expected 3 inputs to download.sh"
    exit 1
fi

##############################################################
#  Assign Inputs
##############################################################

data_url=$1
data_set=$2
data_local=$3

full_url=$data_url/$data_set

##############################################################
#  Download Data
##############################################################

echo "Downloading data from $full_url"

if ! wget -P $data_local --no-check-certificate $full_url; then
    echo "Error.  Download data from $full_url failed"
    exit 1
fi
