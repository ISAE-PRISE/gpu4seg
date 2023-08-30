This read me serves as a guide for correctly running the multi-architecture Segnet evaluation program.

------------------------------------
 Step 1: Download datasets
------------------------------------
- UAVID: https://uavid.nl/
- UDD: https://github.com/MarcWong/UDD

------------------------------------
 Step 2: Verify dataset
------------------------------------
- Verify folders organisation
- UAVID must be:  uavid 
					|-----> uavid_test
					|-----> uavid_train
					|-----> uavid_val
					
- UDD6 must be:  UDD6 
					|-----> metada
					|-----> train
					|-----> val
										
------------------------------------
 Step 3: Create image directory
------------------------------------
- Create a directory within this directory where the images to use are located. 
  For example, "mkdir ./images".
- Afterwards, move the testing images you downloaded (e.g., those in uavid_test), to the 
  recently created directory.
					
------------------------------------
 Step 4: Download binary files
------------------------------------
- Download the binary files (data for deep learning inference) from:
	--> UAVID : https://doi.org/10.34849/2K7NOZ
  	--> UDD6  : https://doi.org/10.34849/6EOIV0
- Create a directory (e.g., "mkdir ./data") where folders contaning the binary files will be placed.
 
------------------------------------
 Step 5: Download stb
------------------------------------
- Download stb libraries. These are used for importing and exporting images.
  Git: https://github.com/nothings/stb:
- Place it in this directory (with data and images directory).

------------------------------------
 Step 6: Compile and execute program
------------------------------------
- Open "Makefile"
- Replace "CUDA_PATH ?= /usr/local/cuda-11.4" with the right path and CUDA version.
- Execute program, for example: ./main 630 images/file24-10.png ./data/ ./images/output_segmentation.png
- Arguments: Please, use "./main -h" to understand the program arguments. 
- The output image should be located under the directory given in the third argument ("./images" in our example).

------------------------------------
Notes
------------------------------------
- Dataset: UAVID is used by default, requiring no code change. To use UDD, the following code modification must be done: 
	(1) "bin_dir_path" with the new path in "main.c" 
	(2) "NB_CLASSES" with the number of classes to consider ( in UDD6) in "segnet_archX_implementations.h" being "X" the implementation ID for any implementation to use (e.g., "SegnetImplementationOriginalTraditionalArch1(...)")
	(3) "d_class_tag_val" with the new coloring code in "segnet_archX_implementations.h" being "X" the implementation ID for any implementation to use (e.g., "SegnetImplementationOriginalTraditionalArch1(...)")
- Compiler: Warnings coming from the stb libraries will appear. You can ignore them.
- Images: PNG format is recommened.



Contacts:
# alfonso.mascarenas-gonzalez@isae-supaero.fr
# jean-baptiste.chaudron@isae-supaero.fr


