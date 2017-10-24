#!/bin/bash

echo "----------------------------------------------------------------"
echo "1: Test on a video using the custom helmet cfg and weight files"
echo "2: Test on a given image using the custom helmet cfg and weight files"
read -p "Please enter a corresponding integer value (press -1 to exit): " USR_OPTION


case "$USR_OPTION" in
	-1)
		echo "Exiting...!"
		echo "----------------------------------------------------------------"
		exit 0
		;;
	0)
		echo "Error, enter a positive integer"
		echo "----------------------------------------------------------------"
		exit 1
		;;
	1)
		echo "Copy the following statement or modify accordingly"
		echo "----------------------------------------------------------------"
		echo "./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights helmet_cyclists.MP4"
		echo "----------------------------------------------------------------"
		;;
	2)
		echo "Copy the following statement or modify accordingly"
		echo "----------------------------------------------------------------"
		echo "./darknet detect cfg/yolo-helmet-detect.cfg yolo-helmet_10000.weights data/helmet_and_non.jpg"
		echo "----------------------------------------------------------------"
		;;
	*)
		echo "Error, enter a correct input"
		echo "----------------------------------------------------------------"
		exit 1
		;;
esac
