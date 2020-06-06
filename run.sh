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
data_set=dev-clean.tar.gz

##############################################################
#  Create Local Data Directory
##############################################################

if [ ! -e $data_local ]; then
    mkdir $data_local
fi

##############################################################
#  Download Data
##############################################################

. scripts/download.sh ${data_url} ${data_set} ${data_local}

##############################################################
#  Decompress Data
##############################################################

. scripts/decompress.sh ${data_set} ${data_local}
