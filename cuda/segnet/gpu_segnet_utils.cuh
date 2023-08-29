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

/*----------------------- gpu_segnet_utils.cuh -----------------
|  File gpu_segnet_utils.cuh
|
|  Description: CUDA kernel declarations of operations required  
|  		by the Segnet architecture
|
|  Version: 1.0
*-----------------------------------------------------------------------------------*/
 



/* resizeImgCUDA
 *
 * Description: CUDA kernel for resizing an image using the Nearest Neighbor Interpolation technique 
 *		The threads move across the output image.
 *
 * Parameter:   
 *		- uint8_t *imgf: Resized matrix of the image
 *		- unsigned int Nx: imgf width (output image)
 *		- unsigned int Ny: imgf height (output image)
 *		- uint8_t *img: Input matrix of image
 *		- unsigned int x_img: img width (input image)
 *		- unsigned int channels_img: Number of channels
 *		- float x_conv: Ratio between the input and output img horizontal size 
 *		- float y_conv: Ratio between the input and output img vertical size 
 *
 * Returns:     Nothing
 *
 * */
__global__ void resizeImgCUDA(uint8_t *imgf, unsigned Nx, unsigned Ny, uint8_t *img, unsigned int x_img, unsigned channels_img, float x_conv, float y_conv){

  	// Block index
  	int bx = blockIdx.x;

  	// Thread index
  	int tx = threadIdx.x;
  	
  	// Thread index
  	int idx = bx*blockDim.x + tx;
	unsigned temp = idx;
	
	if(idx < Nx*Ny*channels_img){
		
	    	int z_pix = idx % channels_img;
	    	temp = idx / channels_img;
	    	int x_pix = temp % Nx;
		temp = temp / Nx; 	
	    	int y_pix = temp;

	  	unsigned xz_off_out = Nx*channels_img;
	  	unsigned xz_off_in = x_img*channels_img;
	  
	  	// Across the output image horizontal 
		imgf[(y_pix*xz_off_out) + x_pix*channels_img + z_pix] = img[((unsigned)round(y_pix*y_conv))*xz_off_in + ((unsigned)round(x_pix*x_conv))*channels_img + z_pix];  

  	}
}


/* createRGBclassesCUDA
 *
 * Description: CUDA kernel for creating an RGB image as function of 
 *		the dominant class of each input matrix pixel.
 *
 * Parameter:   
 *		- uint8_t *out_rgb_arr: Colored image
 *		- unsigned int count: Number of pixels belonging to a specific class (WARNING: currently not used)
 *		- uint8_t *in_classes_arr: Input matrix with dominant class per pixel
 *		- unsigned int out_width: Width of the output matrix
 *		- unsigned int out_height: Height of the output matrix
 *		- uint8_t r_val: Value for the red channel of the output matrix 
 *		- uint8_t g_val: Value for the green channel of the output matrix 
 *		- uint8_t b_val: Value for the blue channel of the output matrix 
 *
 * Note: 	out_height = in_height, out_width = in_width
 *
 * Returns:     Nothing
 *
 * */
__global__ void createRGBclassesCUDA(uint8_t *out_rgb_arr, unsigned* count, uint8_t *in_classes_arr, unsigned out_width, unsigned out_height, unsigned out_channels, unsigned class_tag, uint8_t r_val, uint8_t g_val, uint8_t b_val){
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    
    // Input index
    int i = blockDim.x * bx + tx;
    // Output index
    int j = out_channels*i;
    
    unsigned numElements = out_width*out_height;
    
    // Reset counter
    count[class_tag] = 0;
      		
    if (i < numElements){
    	if(in_classes_arr[i] == class_tag){
    		out_rgb_arr[j + 0] = r_val;  
    		out_rgb_arr[j + 1] = g_val;  
    		out_rgb_arr[j + 2] = b_val;  
  	 //	count[class_tag]++; // Should be atomic
    	}
    }

}



/* indicesUnpooling_CUDA
 *
 * Description: CUDA kernel for 3D unpooling using the max values indices during 3D pooling. 
 *		The threads move across the output image. It is assumed that the elements 
 *		of the output image are initialized to zero.
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input matrix of image
 *		- int Nx: img width
 *		- int Ny: img height
 *		- int Nz: img depth
 *		- unsigned *indices: Array with the indices of the max value from maxpooling the img
 *
 * Returns:     Nothing
 *
 * */
__global__ void indicesUnpooling_CUDA(float *imgf, float *img, int Nx, int Ny, int Nz, unsigned *indices){

  	// Block index
  	int bx = blockIdx.x;

  	// Thread index
  	int tx = threadIdx.x;
  	
  	// Thread index
  	int tdx = bx*blockDim.x + tx;
	unsigned temp = tdx;
  	
	if(tdx < Nx*Ny*Nz){
	
	    	int z_pix = tdx % Nz;
	    	temp = tdx / Nz;
	    	int x_pix = temp % Nx;
		temp = temp / Nx; 	
	    	int y_pix = temp;
	    	
		unsigned idx = y_pix*Nx*Nz + x_pix*Nz + z_pix;
		imgf[indices[idx]] = img[idx]; 	
  	}
}

 
 
/* argMax3D_CUDA
 *
 * Description: CUDA kernel for locating the dominant class for each element 
 *		of the matrix.
 *
 * Parameter:   
 *		- uint8_t *output_arr: Colored image
 *		- float *input_arr: Input matrix whose depth contains the values for each considered segmentation class
 *		- unsigned int in_depth: Depth of the output matrix
 *		- unsigned int size: Total size of the output matrix (Width * Height * 1)
 *
 * Returns:     Nothing
 *
 * */   
__global__ void argMax3D_CUDA(uint8_t *output_arr, float *input_arr, unsigned in_depth, unsigned size){
    
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    int i = (bx*blockDim.x) + tx;
    
    if(i < size){
        
        float temp = input_arr[i*in_depth]; 
        uint8_t index = 0;  
    
	for(int cnt_depth = 1; cnt_depth < in_depth; cnt_depth++){
		float next_val = input_arr[i*in_depth + cnt_depth];

		if(temp < next_val){
			temp = next_val;
			index = cnt_depth;
		}

	}
		
    	output_arr[i] = index;
    	
    }

}



/* maxPooling_CUDA
 *
 * Description: CUDA kernel for max downsampling a 3D matrix while recording the indices. 
 *		The threads move across the output image.
 *
 * template:
 *		- int DOWNSAMPLE: Downsampling size  (total size = DOWNSAMPLE*DOWNSAMPLE)
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- unsigned *d_idx: Array with the indices of the max value from the input img
 *		- float *img: Input matrix of image
 *		- int Nx: img width
  *		- int Ny: img height
 *		- int Nz: img depth
 *
 * Returns:     Nothing
 *
 * */
template <int DOWNSAMPLE> __global__ void maxPooling_CUDA(float *imgf, unsigned *d_idx, float *img, unsigned Nx, unsigned Ny, unsigned Nz){

  	// Block index
  	int bx = blockIdx.x;

  	// Thread index
  	int tx = threadIdx.x;

  	// Thread id in grid
	unsigned tdx = tx + bx*blockDim.x;	
	unsigned temp = tdx;

     
	if(tdx < (Nx*Ny*Nz)/(DOWNSAMPLE*DOWNSAMPLE)){

		unsigned z_pix = tdx % Nz;
		temp = tdx / Nz;
		unsigned x_pix = temp % (Nx/DOWNSAMPLE);
		temp = temp / (Nx/DOWNSAMPLE); 	
		unsigned y_pix = temp;
				
	  	// horizontal-depth length of input matrix
	  	unsigned xz_off = Nx*Nz;
			
  	  	// Max value
	  	float max_val = 0;
	  	float cur_val = 0;
	  	int max_index = -1;
	  	int idx = 0;
	  	
  	    	#pragma unroll	
  		for (int i = 0; i<DOWNSAMPLE; i++){
  		    	#pragma unroll	
  			for (int j = 0; j<DOWNSAMPLE; j++){
  				idx = xz_off*(DOWNSAMPLE*y_pix + i) + Nz*(DOWNSAMPLE*x_pix + j) + z_pix;
	  			cur_val = img[idx];
  				
  				if(cur_val > max_val){
  					max_val = cur_val;
  					max_index = idx;
				}
			}
  		}
  		
  		// Write to max value and its respective index to device memory
  		int idx_out = y_pix*(Nx/DOWNSAMPLE)*Nz + x_pix*Nz + z_pix;	

		imgf[idx_out] = max_val;
		
		if(max_index == -1)
			d_idx[idx_out] = idx;  
		else
			d_idx[idx_out] = max_index;  		

  		
  	
  	}
}





