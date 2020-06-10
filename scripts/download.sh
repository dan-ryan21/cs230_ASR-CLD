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
    echo -e "\n*** Error.  Expected 3 inputs to download.sh ***\n"
    exit 1
fi

##############################################################
#  Assign Inputs
##############################################################

data_url=$1
data_set=$2
data_local=$3

tar_file=$data_set.tar.gz
full_url=$data_url/$tar_file

##############################################################
#  Download Data
##############################################################

if [ ! -f $data_local/$tar_file ]; then
    echo -e "\n*** Downloading data from $full_url ***\n"
	if ! wget -P $data_local --no-check-certificate $full_url; then
		echo -e "\n*** Error.  Download data from $full_url failed ***\n"
		exit 1
	fi
else
    echo "\n*** Data already exists ***\n"
fi
