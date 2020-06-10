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

if [ $# -ne 1 ]; then
    echo -e "\n*** Error.  Expected 1 input to convert.sh ***\n"
    exit 1
fi

##############################################################
#  Assign Inputs
##############################################################

data_folder=$1

##############################################################
#  Convert .flac files to .wav
##############################################################

echo -e "\n*** Converting .flac files to .wav format.  This may take a while. ***\n"

for readerFolder in $data_folder/*/; do
    for chapterFolder in $readerFolder*/; do
	    for flacFile in $chapterFolder*.flac; do
		    baseFile="${flacFile%.*}"
			wavFile=$baseFile.wav
			
			if [ ! -f $wavFile ]; then
				ffmpeg -v quiet -i $flacFile $wavFile
			fi
			
			rm -f $flacFile
		done
	done
done
