#! /bin/bash

echo "Executing Segnet architectures... "

# Configured for NVIDIA Jetson AGX Orin
sudo /usr/bin/jetson_clocks --store ./my_jetson_clocks.txt
sudo /usr/bin/jetson_clocks --fan
sudo cat /sys/kernel/debug/bpmp/debug/clk/gpusysmux/max_rate > /sys/kernel/debug/bpmp/debug/clk/gpusysmux/rate


# 480x360 resized image for all Segnet architectures
if [[ $1 == 0 ]]
then
	for config_type in $(seq 0 10 30) 
	do
		for arch_type in {100..900..100}
		do
			for resol_type in {0..0..1} # {0..2..1}
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png
			done
		done
	done
fi	
	
# 640x360, 820x460 resized image for integrated Segnet architectures 1
if [[ $1 == 1 ]]
then
	for config_type in $(seq 10 20 30) 
	do
		for arch_type in {100..100..0}
		do
			for resol_type in {1..2..1} 
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png	 
			done
		done
	done
fi

# 640x360, 820x460 resized image for integrated Segnet architectures 2
if [[ $1 == 2 ]]
then
	for config_type in $(seq 10 20 30) 
	do
		for arch_type in {200..200..0}
		do
			for resol_type in {1..2..1} 
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png	 
			done
		done
	done
fi

# 640x360, 820x460 resized image for integrated Segnet architecture 4
if [[ $1 == 4 ]]
then
	for config_type in $(seq 10 20 30) 
	do
		for arch_type in {400..400..0}
		do
			for resol_type in {1..2..1} 
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png	 
			done
		done
	done
fi

# 640x360, 820x460 resized image for integrated Segnet architecture 6
if [[ $1 == 6 ]]
then
	for config_type in $(seq 10 20 30) 
	do
		for arch_type in {600..600..0}
		do
			for resol_type in {1..2..1} 
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png	 
			done
		done
	done
fi

# 640x360, 820x460 resized image for integrated Segnet architecture 8
if [[ $1 == 8 ]]
then
	for config_type in $(seq 10 20 50) 
	do
		for arch_type in {800..800..0}
		do
			for resol_type in {1..2..1} 
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png	 
			done
		done
	done
fi

# 640x360, 820x460 resized image for integrated Segnet architecture 9
if [[ $1 == 9 ]]
then
	for config_type in $(seq 10 20 30) 
	do
		for arch_type in {900..900..0}
		do
			for resol_type in {1..2..1} 
			do
				sleep 30
				nice -n -20 ./main $(( $arch_type + $config_type + $resol_type)) images/file24-10.png ./data/ ./images/output_segmentation.png	 
			done
		done
	done
fi


sudo /usr/bin/jetson_clocks --restore ./my_jetson_clocks.txt 
sudo cat /sys/kernel/debug/bpmp/debug/clk/gpusysmux/min_rate > /sys/kernel/debug/bpmp/debug/clk/gpusysmux/rate
sudo rm ./my_jetson_clocks.txt
