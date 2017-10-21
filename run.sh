#!/bin/bash

echo "----------------------------------------------------------------"
echo "1: Test on a video using the custom helmet cfg and weight files"
echo "2: Test on a given image using the custom helmet cfg and weight files"
read -p "Please enter a corresponding integer value (press -1 to exit): " USR_OPTION

while true; do

	case "$USR_OPTION" in
		-1)
			echo "Exiting...!"
			exit 0
			;;
		0) 
			echo "Error, enter a positive integer"
			exit 1
			;;
		1)
			./darknet detector demo cfg/coco.data cfg/yolo-helmet-detect.cfg yolo-helmet_10000.weights helmet_cyclists.MP4
			;;
		2)
			./darknet detect cfg/yolo-helmet-detect.cfg yolo-helmet_10000.weights data/helmet_and_non.jpg
			;;
		*)
			echo "Error, enter a correct input"
			exit 1
			;;
	esac

	read -p "Please enter a corresponding integer value (press -1 to exit): " USR_OPTION
done


