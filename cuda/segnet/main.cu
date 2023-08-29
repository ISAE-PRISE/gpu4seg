// ---------------------------------------------------------------------
// GPU4SEG project
// Copyright (C) 2022-2023 ISAE
// 
// Purpose:
// Evaluation of iGPU for semantic segmentation on embedded systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// alfonso.mascarenas-gonzalez@isae-supaero.f
// ---------------------------------------------------------------------

/*--------------------------- main.c -------------------------------------
|  File main.c
|
|  Description:  The evaluation of different Segnet architectures are performed. 
|		   Tested for the Jetson AGX Orin 64 GB.
|
|  Version: 1.0
|
| Contact:
| alfonso.mascarenas-gonzalez@isae-supaero.fr
| jean-baptiste.chaudron@isae-supaero.fr
*-----------------------------------------------------------------------*/

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

// Include the string library
#include <string>

// Open files
#include <fstream>

// Image management libs: https://github.com/nothings/stb
#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"

// Wraps some of the C API routines
#include <cuda_runtime.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

// GPU clock management
#include "gpu_clock_meas.cuh"

// Common Segnet functions
#include "segnet_utils.h"

// Various Segnet GPU functions
#include "gpu_segnet_utils.cuh"

// Traditional convolution GPU functions
#include "gpu_segnet_conv_kernels.cuh"

// Winograd output matrix of 2x2 GPU functions
#include "gpu_segnet_wino2x2_kernels.cuh"


// All Segnet implementations
#include "segnet_arch1_implementations.h"
#include "segnet_arch2_implementations.h"
#include "segnet_arch3_implementations.h"
#include "segnet_arch4_implementations.h"
#include "segnet_arch6_implementations.h"
#include "segnet_arch8_implementations.h"
#include "segnet_arch9_implementations.h"

/* GLOBAL VARIABLES*/
//#define DEBUG_MODE 


// Test configuration variables
unsigned implementation_option = 100;
char input_img_path[256];
char bin_dir_path[256];
char output_img_path[256];

int main(int argc, char* argv[]){

    // Program arguments processing 
    if (argc == 2 && (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "-help") || checkCmdLineFlag(argc, (const char **)argv, "-h"))){
	printf("***** HELP *****\n");
	printf("Arguments: \n");
	
	printf("(1) Implementation option: Segnet architecture version to use. The first digit indicates the Segnet architecture to use. The second digit is for the internal configuration, where 0 and 1 use traditional convolutions, 2 and 3 use Winograd convolutions with an output tile of 2x2, and 4 and 5 use Winograd convolutions with an output tile of 4x4. The third digit indicates the image input resolution for Segnet, where 0 is 480x360, 1 is 640x360 and 2 is 820x460. Not all combinations have been coded. The followings are accepted:\n");
	printf("\t 100 101 102 110 111 112 120 130 131 132 200 210 211 212 220 230 231 232 300 310 320 330 400 410 411 412 420 430 431 432 600 610 611 612 620 630 631 632 800 810 811 812 820 830 831 832 900 910 911 912 920 930 931 932 \n");
	
	printf("(2) Input image path: The path and name of the image to use for the segmentation, e.g., ./images/file24-10.png \n");
    	printf("(3) Binary files path: The directory where all Segnet architecture binary files are located, e.g., ./data/\n\n");
    	printf("(4) Output image path: The path and name of the segmented image to save, e.g., ./images/output_segmentation.png\n\n");
    	
    	printf("Example: ./main 630 images/file24-10.png ./data/ ./images/output_segmentation.png \n");
    	
    	return -100;
    }
    else if (argc == 5) {
        implementation_option = std::atoi(argv[1]);
  	strcpy(input_img_path, argv[2]); 
  	strcpy(bin_dir_path, argv[3]); 
  	strcpy(output_img_path, argv[4]); 

    }
    else{
    	printf("Program arguments error. Type './main -h' for more information. \n");
    	return -1;
    }
    
    /*************************/
    /* Segnet architecture 1 */
    /*************************/
    
    if (implementation_option == 100){
    	printf("SegnetImplementationOriginalTraditionalArch1 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet1_uavid_480_360/");

    	SegnetImplementationOriginalTraditionalArch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 101){
    	printf("SegnetImplementationOriginalTraditionalArch1 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet1_uavid_640_360/");
	
    	SegnetImplementationOriginalTraditionalArch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 102){
    	printf("SegnetImplementationOriginalTraditionalArch1 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet1_uavid_820_460/");

    	SegnetImplementationOriginalTraditionalArch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 110){
    	printf("SegnetImplementationIntegratedTraditionalArch1 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet1_uavid_480_360/");
	
    	SegnetImplementationIntegratedTraditionalArch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 111){
    	printf("SegnetImplementationIntegratedTraditionalArch1 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet1_uavid_640_360/");

    	SegnetImplementationIntegratedTraditionalArch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 112){
    	printf("SegnetImplementationIntegratedTraditionalArch1 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

    	strcat(bin_dir_path, "segnet1_uavid_820_460/");
    	
    	SegnetImplementationIntegratedTraditionalArch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }

    else if (implementation_option == 120){  
	printf("SegnetImplementationOriginalWinograd2x2Arch1 - 480x360\n");
        	 
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet1_uavid_480_360/");
	
    	SegnetImplementationOriginalWinograd2x2Arch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
    	
    }

    else if (implementation_option == 130){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch1 - 480x360\n");
    	
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet1_uavid_480_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 131){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch1 - 640x360\n");
    	
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet1_uavid_640_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 132){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch1 - 820x460\n");
    	
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

    	strcat(bin_dir_path, "segnet1_uavid_820_460/");
    	
    	SegnetImplementationIntegratedWinograd2x2Arch1(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    
    /*************************/
    /* Segnet architecture 2 */
    /*************************/
    
    else if (implementation_option == 200){
    	printf("SegnetImplementationOriginalTraditionalArch2 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

   	strcat(bin_dir_path, "segnet2_uavid_480_360/");
   	
    	SegnetImplementationOriginalTraditionalArch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 210){
    	printf("SegnetImplementationIntegratedTraditionalArch2 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet2_uavid_480_360/");

    	SegnetImplementationIntegratedTraditionalArch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 211){
    	printf("SegnetImplementationIntegratedTraditionalArch2 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet2_uavid_640_360/");

    	SegnetImplementationIntegratedTraditionalArch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 212){
    	printf("SegnetImplementationIntegratedTraditionalArch2 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet2_uavid_820_460/");

    	SegnetImplementationIntegratedTraditionalArch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 220){
    	printf("SegnetImplementationOriginalWinograd2x2Arch2 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet2_uavid_480_360/");

    	SegnetImplementationOriginalWinograd2x2Arch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 230){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch2 - 480x360\n");
    	
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet2_uavid_480_360/");
	
    	SegnetImplementationIntegratedWinograd2x2Arch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 231){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch2 - 640x360\n");
    	
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

    	strcat(bin_dir_path, "segnet2_uavid_640_360/");
    	
    	SegnetImplementationIntegratedWinograd2x2Arch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 232){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch2 - 820x460\n");
    	
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet2_uavid_820_460/");

    	SegnetImplementationIntegratedWinograd2x2Arch2(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    /*************************/
    /* Segnet architecture 3 */
    /*************************/
    
    else if (implementation_option == 300){
    	printf("SegnetImplementationOriginalTraditionalArch3 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet3_uavid_480_360/");
	
    	SegnetImplementationOriginalTraditionalArch3(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 310){
    	printf("SegnetImplementationIntegratedTraditionalArch3 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet3_uavid_480_360/");

    	SegnetImplementationIntegratedTraditionalArch3(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 320){
    	printf("SegnetImplementationOriginalWinograd2x2Arch3 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet3_uavid_480_360/");

    	SegnetImplementationOriginalWinograd2x2Arch3(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 330){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch3 - 480x360\n");
    	
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet3_uavid_480_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch3(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    /*************************/
    /* Segnet architecture 4 */
    /*************************/
    
    else if (implementation_option == 400){
    	printf("SegnetImplementationOriginalTraditionalArch4 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet4_uavid_480_360/");

    	SegnetImplementationOriginalTraditionalArch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 410){
    	printf("SegnetImplementationIntegratedTraditionalArch4 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet4_uavid_480_360/");

    	SegnetImplementationIntegratedTraditionalArch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 411){
    	printf("SegnetImplementationIntegratedTraditionalArch4 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet4_uavid_640_360/");

    	SegnetImplementationIntegratedTraditionalArch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 412){
    	printf("SegnetImplementationIntegratedTraditionalArch4 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet4_uavid_820_460/");

    	SegnetImplementationIntegratedTraditionalArch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
   else if (implementation_option == 420){
    	printf("SegnetImplementationOriginalWinograd2x2Arch4 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet4_uavid_480_360/");

    	SegnetImplementationOriginalWinograd2x2Arch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
   else if (implementation_option == 430){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch4 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet4_uavid_480_360/");
	
    	SegnetImplementationIntegratedWinograd2x2Arch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
   else if (implementation_option == 431){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch4 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet4_uavid_640_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 432){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch4 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet4_uavid_820_460/");

    	SegnetImplementationIntegratedWinograd2x2Arch4(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    /*************************/
    /* Segnet architecture 6 */
    /*************************/
    
    else if (implementation_option == 600){
    	printf("SegnetImplementationOriginalTraditionalArch6 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet6_uavid_480_360/");
	
    	SegnetImplementationOriginalTraditionalArch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 610){
    	printf("SegnetImplementationIntegratedTraditionalArch6 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet6_uavid_480_360/");
	
    	SegnetImplementationIntegratedTraditionalArch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 611){
    	printf("SegnetImplementationIntegratedTraditionalArch6 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet6_uavid_640_360/");

    	SegnetImplementationIntegratedTraditionalArch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 612){
    	printf("SegnetImplementationIntegratedTraditionalArch6 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet6_uavid_820_460/");

    	SegnetImplementationIntegratedTraditionalArch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 620){
    	printf("SegnetImplementationOriginalWinograd2x2Arch6 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet6_uavid_480_360/");
	
    	SegnetImplementationOriginalWinograd2x2Arch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 630){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch6 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet6_uavid_480_360/");
	
    	SegnetImplementationIntegratedWinograd2x2Arch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 631){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch6 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet6_uavid_640_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 632){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch6 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet6_uavid_820_460/");

    	SegnetImplementationIntegratedWinograd2x2Arch6(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    /*************************/
    /* Segnet architecture 8 */
    /*************************/
    
    else if (implementation_option == 800){
    	printf("SegnetImplementationOriginalTraditionalArch8 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet8_uavid_480_360/");
	
    	SegnetImplementationOriginalTraditionalArch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 810){
    	printf("SegnetImplementationIntegratedTraditionalArch8 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet8_uavid_480_360/");

    	SegnetImplementationIntegratedTraditionalArch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 811){
    	printf("SegnetImplementationIntegratedTraditionalArch8 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet8_uavid_640_360/");

    	SegnetImplementationIntegratedTraditionalArch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 812){
    	printf("SegnetImplementationIntegratedTraditionalArch8 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet8_uavid_820_460/");

    	SegnetImplementationIntegratedTraditionalArch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
   else if (implementation_option == 820){
    	printf("SegnetImplementationOriginalWinograd2x2Arch8 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet8_uavid_480_360/");

    	SegnetImplementationOriginalWinograd2x2Arch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
   else if (implementation_option == 830){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch8 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet8_uavid_480_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
   else if (implementation_option == 831){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch8 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet8_uavid_640_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 832){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch8 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet8_uavid_820_460/");

    	SegnetImplementationIntegratedWinograd2x2Arch8(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    /*************************/
    /* Segnet architecture 9 */
    /*************************/
    
    else if (implementation_option == 900){
    	printf("SegnetImplementationOriginalTraditionalArch9 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);
    	
	strcat(bin_dir_path, "segnet9_uavid_480_360/");

    	SegnetImplementationOriginalTraditionalArch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 910){
    	printf("SegnetImplementationIntegratedTraditionalArch9 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet9_uavid_480_360/");

    	SegnetImplementationIntegratedTraditionalArch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 911){
    	printf("SegnetImplementationIntegratedTraditionalArch9 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet9_uavid_640_360/");

    	SegnetImplementationIntegratedTraditionalArch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 912){
    	printf("SegnetImplementationIntegratedTraditionalArch9 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);
    	
    	strcat(bin_dir_path, "segnet9_uavid_820_460/");

    	SegnetImplementationIntegratedTraditionalArch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 920){
    	printf("SegnetImplementationOriginalWinograd2x2Arch9 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet9_uavid_480_360/");

    	SegnetImplementationOriginalWinograd2x2Arch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }    
    
    else if (implementation_option == 930){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch9 - 480x360\n");
    
	dim3 img_dim(480, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet9_uavid_480_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);
	
    }
    
    else if (implementation_option == 931){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch9 - 640x360\n");
    
	dim3 img_dim(640, 360, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet9_uavid_640_360/");

    	SegnetImplementationIntegratedWinograd2x2Arch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    
    else if (implementation_option == 932){
    	printf("SegnetImplementationIntegratedWinograd2x2Arch9 - 820x460\n");
    
	dim3 img_dim(820, 460, 1);
    	dim3 filter_dim(3, 3, 1);

	strcat(bin_dir_path, "segnet9_uavid_820_460/");

    	SegnetImplementationIntegratedWinograd2x2Arch9(img_dim, filter_dim, input_img_path, bin_dir_path, output_img_path);

    }
    

}




