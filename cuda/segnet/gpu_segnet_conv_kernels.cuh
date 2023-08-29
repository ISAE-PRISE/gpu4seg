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

/*----------------------- gpu_segnet_conv_kernels.cuh -----------------
|  File gpu_segnet_conv_kernels.cuh
|
|  Description: CUDA kernel declarations of the traditional convolutions,  
|  		with and without batch normalization included in
|		the filter weights.
|
|  Version: 1.0
*-----------------------------------------------------------------------------------*/
 
 
/********************************************/
/* Functions performing batch normalization */
/********************************************/




/* convB_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing traditional convolution.  
 *		It normalizes the input image, which is defined as 8 bits   
 *		integer, by 255. The threads move across the output image.
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the X axis of the filter
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- uint8_t *img: Non-normilized input image
 *		- float *kernel: Matrix with the kernels (filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *mean: Mean value for batch normalization
 *		- float *var: Variance value for batch normalization
 *		- float *weight: Weight value for batch normalization
 *		- float *bias: Bias value for batch normalization
 *		- const float EPSILON: Tiny value added to var for avoiding dividing by zero
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int KERNEL_SIZE> __global__ void convB_norm_ReLu_CUDA(float *imgf, uint8_t *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	
	unsigned idx = tx + bx*blockDim.x;	
	unsigned temp = idx;
	
	if(idx < Nx*Ny*out_ch_offset){
		
	    	int z_pix = idx % out_ch_offset;
	    	temp = idx / out_ch_offset;
	    	int x_pix = temp % Nx;
		temp = temp / Nx; 	
	    	int y_pix = temp;
	    	
		// Lock data into the L1 cache
		 __shared__ float sdata[BLOCK_SIZE_X]; 
 		 // Add convolution bias
		 sdata[tx] = 0;
		__syncthreads();
	 	
		unsigned center = (KERNEL_SIZE - 1)/2;
		unsigned xz_img_off = Nx*Nz;
		unsigned xz_ker_off = KERNEL_SIZE*Nz*out_ch_offset;
		
		uint8_t img_val = 0;
		int ii, jj;
	
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - center);
			
			if((jj >= 0) && (jj <= Ny-1)){
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - center);
				      
				      if ((ii >= 0) && (ii <= Nx-1)){
				      unsigned kernel_idx = kj*xz_ker_off + ki*out_ch_offset*Nz;  
				      
				      	 // Across the filter depth 
					for (int kk = 0; kk<Nz; kk++){
					     img_val = (img[jj*xz_img_off + ii*Nz + kk]);
					     sdata[tx] += img_val * kernel[kernel_idx + kk*out_ch_offset + z_pix];
					}
									    
				     }
				  }
			  }
		 }	
			
		__syncthreads();
				 		   
		 // Calculate index
		 int id_pix = y_pix*Nx*out_ch_offset + x_pix*out_ch_offset + z_pix;		   	 
		 // Batch normalization
	 	 sdata[tx] = ((sdata[tx]/255 - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);
	  	 // ReLu (max(0,x))
	 	 imgf[id_pix] = (sdata[tx] < 0) ? 0 : sdata[tx];	
 	 
 	 
 	 }		
			  
}


/* convB_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing traditional convolution.  
 *		The input image is defined as a float.  
 *		The threads move across the output image.
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the X axis of the filter
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the kernels (filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *mean: Mean value for batch normalization
 *		- float *var: Variance value for batch normalization
 *		- float *weight: Weight value for batch normalization
 *		- float *bias: Bias value for batch normalization
 *		- const float EPSILON: Tiny value added to var for avoiding dividing by zero
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int KERNEL_SIZE> __global__ void convB_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	
	unsigned idx = tx + bx*blockDim.x;	
	unsigned temp = idx;
	
	if(idx < Nx*Ny*out_ch_offset){
		
		unsigned z_pix = idx % out_ch_offset;
		temp = (idx / out_ch_offset);
		unsigned x_pix = temp % Nx;
		temp = (temp / Nx); 	
		unsigned y_pix = temp;

		// Lock data into the L1 cache
		 __shared__ float sdata[BLOCK_SIZE_X]; 
 		 // Add convolution bias
		 sdata[tx] = 0;
		__syncthreads();
		 
		unsigned center = (KERNEL_SIZE - 1)/2;
		unsigned xz_img_off = Nx*Nz;
		unsigned xz_ker_off = KERNEL_SIZE*Nz*out_ch_offset;

		float img_val = 0;
		int ii, jj;
			
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - center);
			
			if((jj >= 0) && (jj <= Ny-1)){
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - center);
				      
				      if ((ii >= 0) && (ii <= Nx-1)){
				      unsigned kernel_idx = kj*xz_ker_off + ki*out_ch_offset*Nz;  
				      
				      	 // Across the filter depth 
					for (int kk = 0; kk<Nz; kk++){
					     img_val = img[jj*xz_img_off + ii*Nz + kk];
					     sdata[tx] += img_val * kernel[kernel_idx + kk*out_ch_offset + z_pix];
					}
									    
				     }
				  }
			  }
		 }	
		 
		 // Batch normalization
	 	 sdata[tx] = ((sdata[tx] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);
 		 // Calculate index
		 int id_pix = y_pix*Nx*out_ch_offset + x_pix*out_ch_offset + z_pix;	
	  	 // ReLu (max(0,x))
	 	 imgf[id_pix] = (sdata[tx] < 0) ? 0 : sdata[tx];
	 	 
	}	  

	
}


/* convA_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing traditional convolution.    
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the X axis of the filter
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the kernels (filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *mean: Mean value for batch normalization
 *		- float *var: Variance value for batch normalization
 *		- float *weight: Weight value for batch normalization
 *		- float *bias: Bias value for batch normalization
 *		- const float EPSILON: Tiny value added to var for avoiding dividing by zero
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int KERNEL_SIZE, const int IMG_DEPTH> __global__ void convA_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	
	unsigned idx = tx + bx*blockDim.x;	
	unsigned temp = idx;
	
	if(idx < Nx*Ny*out_ch_offset){
		unsigned z_pix = idx % out_ch_offset;
		temp = idx / out_ch_offset;
		unsigned x_pix = temp % Nx;
		temp = temp / Nx; 	
		unsigned y_pix = temp;

		// Lock data into the L1 cache
		 __shared__ float sdata[BLOCK_SIZE_X];
		 __shared__ float sImg[IMG_DEPTH]; 
		  
 		 // Add convolution bias
		 sdata[tx] = 0;
		__syncthreads();
		 
		unsigned center = (KERNEL_SIZE - 1)/2;
		unsigned xz_img_off = Nx*Nz;
		unsigned xz_ker_off = KERNEL_SIZE*Nz*out_ch_offset;

		int ii, jj;
	

		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - center);
			
			if((jj >= 0) && (jj <= Ny-1)){
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - center);

				      if ((ii >= 0) && (ii <= Nx-1)){
				      unsigned kernel_idx = kj*xz_ker_off + ki*out_ch_offset*Nz;  

	
					for (int cnt = tx; cnt<Nz; cnt+=blockDim.x)
						sImg[cnt] = img[jj*xz_img_off + ii*Nz + cnt];

					
					__syncthreads();

				      	 // Across the filter depth 
					for (int kk = 0; kk<Nz; kk++)
						sdata[tx] += sImg[kk] * kernel[kernel_idx + kk*out_ch_offset + z_pix];
		
						__syncthreads();

			    
				     }
				  }
			  }
		 }	
	   	 
		 // Batch normalization
	 	 sdata[tx] = ((sdata[tx] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);
 		 // Calculate index
		 int id_pix = y_pix*Nx*out_ch_offset + x_pix*out_ch_offset + z_pix;	
	  	 // ReLu (max(0,x))
	 	 imgf[id_pix] = (sdata[tx] < 0) ? 0 : sdata[tx];
	 	 


	}	  

	
}





/*******************************************************************************/
/* Functions assuming that batch normalization is included in the filter weghts*/
/*******************************************************************************/




/* convB_ReLu_CUDA
 *
 * Description: CUDA kernel for performing traditional convolution.  
 *		It normalizes the input image, which is defined as 8 bits   
 *		integer, by 255. The threads move across the output image.
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the X axis of the filter
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- uint8_t *img: Non-normilized input image
 *		- float *kernel: Matrix with the kernels (filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int KERNEL_SIZE> __global__ void convB_ReLu_CUDA(float *imgf, uint8_t *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	
	unsigned idx = tx + bx*blockDim.x;	
	unsigned temp = idx;
	
	if(idx < Nx*Ny*out_ch_offset){
		
	    	int z_pix = idx % out_ch_offset;
	    	temp = idx / out_ch_offset;
	    	int x_pix = temp % Nx;
		temp = temp / Nx; 	
	    	int y_pix = temp;
	    	
		// Lock data into the L1 cache
		 __shared__ float sdata[BLOCK_SIZE_X]; 
 		 // Add convolution bias
		 sdata[tx] = *(conv_bias + z_pix);
		__syncthreads();
	 	
		const unsigned CENTER = (KERNEL_SIZE - 1)/2;
		unsigned xz_img_off = Nx*Nz;
		unsigned xz_ker_off = KERNEL_SIZE*Nz*out_ch_offset;
		
		uint8_t img_val = 0;
		int ii, jj;
	
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - CENTER);
				      
				      if ((ii >= 0) && (ii <= Nx-1)){
				      unsigned kernel_idx = kj*xz_ker_off + ki*out_ch_offset*Nz;  
				      
				      	 // Across the filter depth 
					for (int kk = 0; kk<Nz; kk++){
     					     img_val = img[jj*xz_img_off + ii*Nz + kk];
					     sdata[tx] += img_val * kernel[kernel_idx + kk*out_ch_offset + z_pix];
					}
					sdata[tx] = sdata[tx];
									    
				     }
				  }
			  }
		 }	
			
		__syncthreads();
				 		   
		 // Calculate index
		 int id_pix = y_pix*Nx*out_ch_offset + x_pix*out_ch_offset + z_pix;		   	 

	  	 // ReLu (max(0,x))
	 	 imgf[id_pix] = (sdata[tx] < 0) ? 0 : sdata[tx]/255;	
 	 
 	 
 	 }		
			  
}



/* convB_ReLu_CUDA
 *
 * Description: CUDA kernel for performing traditional convolution.  
 *		The input image is defined as a float.  
 *		The threads move across the output image.
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the X axis of the filter
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the kernels (filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int KERNEL_SIZE> __global__ void convB_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	
	unsigned idx = tx + bx*blockDim.x;	
	unsigned temp = idx;
	
	if(idx < Nx*Ny*out_ch_offset){
		
		unsigned z_pix = idx % out_ch_offset;
		temp = idx / out_ch_offset;
		unsigned x_pix = temp % Nx;
		temp = temp / Nx; 	
		unsigned y_pix = temp;

		// Lock data into the L1 cache
		 __shared__ float sdata[BLOCK_SIZE_X]; 
 		 // Add convolution bias
		 sdata[tx] = *(conv_bias + z_pix);
		__syncthreads();
		 
		unsigned center = (KERNEL_SIZE - 1)/2;
		unsigned xz_img_off = Nx*Nz;
		unsigned xz_ker_off = KERNEL_SIZE*Nz*out_ch_offset;

		float img_val = 0;
		int ii, jj;
			
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - center);
			
			if((jj >= 0) && (jj <= Ny-1)){
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - center);
				      
				      if ((ii >= 0) && (ii <= Nx-1)){
				      unsigned kernel_idx = kj*xz_ker_off + ki*out_ch_offset*Nz;  
				      
				      	 // Across the filter depth 
					for (int kk = 0; kk<Nz; kk++){
					     img_val = img[jj*xz_img_off + ii*Nz + kk];
					     sdata[tx] += img_val * kernel[kernel_idx + kk*out_ch_offset + z_pix];
					}
									    
				     }
				  }
			  }
		 }	
				 		   

 		 // Calculate index
		 int id_pix = y_pix*Nx*out_ch_offset + x_pix*out_ch_offset + z_pix;	
	  	 // ReLu (max(0,x))
	 	 imgf[id_pix] = (sdata[tx] < 0) ? 0 : sdata[tx];		
		
	}	  

	
}


/* convA_ReLu_CUDA
 *
 * Description: CUDA kernel for performing traditional convolution.    
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the X axis of the filter
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the kernels (filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int KERNEL_SIZE, const int IMG_DEPTH> __global__ void convA_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	
	unsigned idx = tx + bx*blockDim.x;	
	unsigned temp = idx;
	
	if(idx < Nx*Ny*out_ch_offset){
		unsigned z_pix = idx % out_ch_offset;
		temp = idx / out_ch_offset;
		unsigned x_pix = temp % Nx;
		temp = temp / Nx; 	
		unsigned y_pix = temp;

		// Lock data into the L1 cache
		 __shared__ float sdata[BLOCK_SIZE_X];
		 __shared__ float sImg[IMG_DEPTH]; 
		  
 		 // Add convolution bias
		 sdata[tx] = *(conv_bias + z_pix);
		__syncthreads();
		 
		unsigned center = (KERNEL_SIZE - 1)/2;
		unsigned xz_img_off = Nx*Nz;
		unsigned xz_ker_off = KERNEL_SIZE*Nz*out_ch_offset;

		int ii, jj;	

		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - center);
			
			if((jj >= 0) && (jj <= Ny-1)){
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - center);

				      if ((ii >= 0) && (ii <= Nx-1)){
				      unsigned kernel_idx = kj*xz_ker_off + ki*out_ch_offset*Nz;  

	
					for (int cnt = tx; cnt<Nz; cnt+=blockDim.x)
						sImg[cnt] = img[jj*xz_img_off + ii*Nz + cnt];

					
					__syncthreads();

				      	 // Across the filter depth 
					for (int kk = 0; kk<Nz; kk++)
						sdata[tx] += sImg[kk] * kernel[kernel_idx + kk*out_ch_offset + z_pix];

						__syncthreads();
					
			    
				     }
				  }
			  }
		 }	
 
 		 // Calculate index
		 int id_pix = y_pix*Nx*out_ch_offset + x_pix*out_ch_offset + z_pix;	
	  	 // ReLu (max(0,x))
	 	 imgf[id_pix] = (sdata[tx] < 0) ? 0 : sdata[tx];

	}	  

	
}

