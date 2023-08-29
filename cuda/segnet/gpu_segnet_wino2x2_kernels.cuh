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

/*----------------------- gpu_segnet_wino2x2_kernels.cuh -----------------
|  File gpu_segnet_wino2x2_kernels.cuh
|
|  Description: CUDA kernel declarations of the 2x2 Winograd convolutions,  
|  		with and without batch normalization included in
|		the filter weights.
|
|  Version: 1.0
*-----------------------------------------------------------------------------------*/
 
/* get_winograd2x2_input_tile_transformation
 *
 * Description: Performs the input tile transformation according to a 2x2 output
 *		tile and a 3x3 filter configuration (V = BT * d * B).
 *
 * Parameter:   
 *		- float *V: Resultant transformed input tile
 *		- float *d: Input tile of the feature to be transformed
 *
 * Returns:     Nothing
 *
 * */
inline __device__ void get_winograd2x2_input_tile_transformation(float *V, float *d){
	V[0] = (d[0]-d[8])-(d[2]-d[10]);       
	V[1] = (d[1]-d[9])+(d[2]-d[10]); 
	V[2] = -(d[1]-d[9])+(d[2]-d[10]); 
	V[3] = (d[1]-d[9])-(d[3]-d[11]);
	
	V[4] = (d[4]+d[8])-(d[6]+d[10]);       
	V[5] = (d[5]+d[9])+(d[6]+d[10]); 
	V[6] = -(d[5]+d[9])+(d[6]+d[10]); 
	V[7] = (d[5]+d[9])-(d[7]+d[11]);  
	
	V[8] = (-d[4]+d[8])-(-d[6]+d[10]);       
	V[9] = (-d[5]+d[9])+(-d[6]+d[10]); 
	V[10] = -(-d[5]+d[9])+(-d[6]+d[10]); 
	V[11] = (-d[5]+d[9])-(-d[7]+d[11]); 
	
	V[12] = (d[4]-d[12])-(d[6]-d[14]);       
	V[13] = (d[5]-d[13])+(d[6]-d[14]); 
	V[14] = -(d[5]-d[13])+(d[6]-d[14]); 
	V[15] = (d[5]-d[13])-(d[7]-d[15]); 

}
 
 
/* get_winograd2x2_output_tile_transformation
 *
 * Description: Performs the output tile transformation according to a 2x2 output
 *		tile and a 3x3 filter configuration. (Y = AT * M * A)
 *
 * Parameter:   
 *		- float *V: Resultant transformed input tile
 *		- float *M: Input tile of the feature to be transformed
 *		- const unsigned BLOCK_SIZE_X: Threads per block of the compute kernel
 *		- unsigned tx: Thread id
 *
 * Returns:     Nothing
 *
 * */
inline __device__ void get_winograd2x2_output_tile_transformation(float *Y, float *M, const unsigned BLOCK_SIZE_X, unsigned tx){ 

	Y[0] = (M[0*BLOCK_SIZE_X + tx] + M[4*BLOCK_SIZE_X + tx] + M[8*BLOCK_SIZE_X + tx]) + (M[1*BLOCK_SIZE_X + tx] + M[5*BLOCK_SIZE_X + tx] + M[9*BLOCK_SIZE_X + tx]) + (M[2*BLOCK_SIZE_X + tx] + M[6*BLOCK_SIZE_X + tx] + M[10*BLOCK_SIZE_X + tx]);
	Y[1] = (M[1*BLOCK_SIZE_X + tx] + M[5*BLOCK_SIZE_X + tx] + M[9*BLOCK_SIZE_X + tx]) - (M[2*BLOCK_SIZE_X + tx] + M[6*BLOCK_SIZE_X + tx] + M[10*BLOCK_SIZE_X + tx]) - (M[3*BLOCK_SIZE_X + tx] + M[7*BLOCK_SIZE_X + tx] + M[11*BLOCK_SIZE_X + tx]);
	Y[2] = (M[4*BLOCK_SIZE_X + tx] - M[8*BLOCK_SIZE_X + tx] - M[12*BLOCK_SIZE_X + tx]) + (M[5*BLOCK_SIZE_X + tx] - M[9*BLOCK_SIZE_X + tx] - M[13*BLOCK_SIZE_X + tx]) + (M[6*BLOCK_SIZE_X + tx] - M[10*BLOCK_SIZE_X + tx] - M[14*BLOCK_SIZE_X + tx]);
	Y[3] = (M[5*BLOCK_SIZE_X + tx] - M[9*BLOCK_SIZE_X + tx] - M[13*BLOCK_SIZE_X + tx]) - (M[6*BLOCK_SIZE_X + tx] - M[10*BLOCK_SIZE_X + tx] - M[14*BLOCK_SIZE_X + tx]) - (M[7*BLOCK_SIZE_X + tx] - M[11*BLOCK_SIZE_X + tx] - M[15*BLOCK_SIZE_X + tx]);

}

 
 
/********************************************/
/* Functions performing batch normalization */
/********************************************/



/* wino_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2.
 *		It normalizes the input image, which is defined as an 8 bits   
 *		integer, by 255. The threads move across the output image.
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- uint8_t *img: Non-normilized input image
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino_conv2x2_norm_ReLu_CUDA(float *imgf, uint8_t *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;
	

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ int sImg[IMG_DEPTH]; 

	int ii, jj;

	int in_tiles[16][IMG_DEPTH];

	// Reset values for padding computations
	#pragma unroll
	for (int cnt = 0; cnt<TILE_SIZE; cnt++){
		in_tiles[cnt][0] = 0;
		in_tiles[cnt][1] = 0;
		in_tiles[cnt][2] = 0;
	}

	// Fetch the input tile data considering the padding
	#pragma unroll
	for (int kj = 0; kj<KERNEL_SIZE; kj++){
		jj = kj + (y_pix - CENTER);
		
		if((jj >= 0) && (jj <= Ny-1)){
			#pragma unroll
			for (int ki = 0; ki<KERNEL_SIZE; ki++){
			      ii = ki + (x_pix - CENTER);

			      if ((ii >= 0) && (ii <= Nx-1)){
				      	in_tiles[kj*KERNEL_SIZE + ki][0] =  (img[jj*Nx*Nz + ii*Nz + 0]);
				      	in_tiles[kj*KERNEL_SIZE + ki][1] =  (img[jj*Nx*Nz + ii*Nz + 1]);
				      	in_tiles[kj*KERNEL_SIZE + ki][2] =  (img[jj*Nx*Nz + ii*Nz + 2]);
			      	}

			  }
		  }
	}

	// As many Bds as tx
	float Bd[TILE_SIZE][IMG_DEPTH];
	
	#pragma unroll
	for(int cnt = 0; cnt < IMG_DEPTH; cnt++){
		Bd[0][cnt] = (in_tiles[0][cnt]-in_tiles[8][cnt])-(in_tiles[2][cnt]-in_tiles[10][cnt]),       
		Bd[1][cnt] = (in_tiles[1][cnt]-in_tiles[9][cnt])+(in_tiles[2][cnt]-in_tiles[10][cnt]); 
		Bd[2][cnt] = -(in_tiles[1][cnt]-in_tiles[9][cnt])+(in_tiles[2][cnt]-in_tiles[10][cnt]);
		Bd[3][cnt] = (in_tiles[1][cnt]-in_tiles[9][cnt])-(in_tiles[3][cnt]-in_tiles[11][cnt]);
		
		Bd[4][cnt] = (in_tiles[4][cnt]+in_tiles[8][cnt])-(in_tiles[6][cnt]+in_tiles[10][cnt]);       
		Bd[5][cnt] = (in_tiles[5][cnt]+in_tiles[9][cnt])+(in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[6][cnt] = -(in_tiles[5][cnt]+in_tiles[9][cnt])+(in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[7][cnt] = (in_tiles[5][cnt]+in_tiles[9][cnt])-(in_tiles[7][cnt]+in_tiles[11][cnt]);  
	
		Bd[8][cnt] = (-in_tiles[4][cnt]+in_tiles[8][cnt])-(-in_tiles[6][cnt]+in_tiles[10][cnt]);       
		Bd[9][cnt] = (-in_tiles[5][cnt]+in_tiles[9][cnt])+(-in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[10][cnt] = -(-in_tiles[5][cnt]+in_tiles[9][cnt])+(-in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[11][cnt] = (-in_tiles[5][cnt]+in_tiles[9][cnt])-(-in_tiles[7][cnt]+in_tiles[11][cnt]); 
		
		Bd[12][cnt] = (in_tiles[4][cnt]-in_tiles[12][cnt])-(in_tiles[6][cnt]-in_tiles[14][cnt]);       
		Bd[13][cnt] = (in_tiles[5][cnt]-in_tiles[13][cnt])+(in_tiles[6][cnt]-in_tiles[14][cnt]); 
		Bd[14][cnt] = -(in_tiles[5][cnt]-in_tiles[13][cnt])+(in_tiles[6][cnt]-in_tiles[14][cnt]); 
		Bd[15][cnt] = (in_tiles[5][cnt]-in_tiles[13][cnt])-(in_tiles[7][cnt]-in_tiles[15][cnt]); 
	}
	
	

	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){

		sImg[0] = Bd[jj][0];
		sImg[1] = Bd[jj][1];
		sImg[2] = Bd[jj][2];
		
		sdata[jj*out_ch_offset + tx] = 0;
		__syncthreads();
		
		#pragma unroll
		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*out_ch_offset + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		sdata[jj*out_ch_offset + tx] = sdata[jj*out_ch_offset + tx]/255;
		__syncthreads();
	}
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
 			
}



/* wino_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		equal to the output channels.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Non-normilized input image
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino_conv2x2_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ float sImg[IMG_DEPTH]; 
	
	
	int ii, jj;

	float in_tiles[TILE_SIZE];
	// Reset values for padding computations
	#pragma unroll
	for (int cnt = 0; cnt<TILE_SIZE; cnt++)
		in_tiles[cnt] = 0;

	// Fetch the input tile data considering the padding
	#pragma unroll
	for (int kj = 0; kj<KERNEL_SIZE; kj++){
		jj = kj + (y_pix - CENTER);
		
		if((jj >= 0) && (jj <= Ny-1)){
			#pragma unroll
			for (int ki = 0; ki<KERNEL_SIZE; ki++){
			      ii = ki + (x_pix - CENTER);

			      if ((ii >= 0) && (ii <= Nx-1))
			  	in_tiles[kj*KERNEL_SIZE + ki] = img[jj*Nx*Nz + ii*Nz + tx];

			  }
		  }
	}
	
	float Bd[TILE_SIZE]; 
	get_winograd2x2_input_tile_transformation(Bd, in_tiles);	

	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){
	
		sImg[tx] = Bd[jj];
		sdata[jj*out_ch_offset + tx] = 0;
		__syncthreads();

		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*out_ch_offset + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		__syncthreads();
	}
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
 			  
}



/* wino2_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		lower than the output channels.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino2_conv2x2_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	


	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; // BLOCK_SIZE_X should be equal to the image depth
	__shared__ float sImg[IMG_DEPTH]; 
	float in_tiles[TILE_SIZE];


	unsigned xz_img_off = Nx*Nz;

	int ii, jj;


	// Reset values for padding computations
	#pragma unroll
	for (int cnt = 0; cnt<TILE_SIZE; cnt++)
		in_tiles[cnt] = 0;

	// Fetch the input tile data considering the padding
	#pragma unroll
	for (int kj = 0; kj<KERNEL_SIZE; kj++){
		jj = kj + (y_pix - CENTER);
		
		if((jj >= 0) && (jj <= Ny-1)){
			#pragma unroll
			for (int ki = 0; ki<KERNEL_SIZE; ki++){
			      ii = ki + (x_pix - CENTER);

			      if ((ii >= 0) && (ii <= Nx-1))
			  	in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx];

			     
			  }
		  }
	}
	
	float Bd[TILE_SIZE]; 
	get_winograd2x2_input_tile_transformation(Bd, in_tiles);


	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){
	
		sImg[tx] = Bd[jj]; 
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		__syncthreads();

		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		__syncthreads();
	}
			
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}		
	
}


/* wino2b_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		lower than the output channels but the former is not multiple 
 *		of the latter.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino2b_conv2x2_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ float sImg[IMG_DEPTH]; 
	float in_tiles[16];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	if(tx < IMG_DEPTH){
		// Reset values for padding computations
		#pragma unroll
		for (int cnt = 0; cnt<TILE_SIZE; cnt++)
			in_tiles[cnt] = 0;

		// Fetch the input tile data considering the padding
		#pragma unroll
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				#pragma unroll
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - CENTER);

				      if ((ii >= 0) && (ii <= Nx-1))
				  	in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx];

				     
				  }
			  }
		}
	}
	
	float Bd[TILE_SIZE]; 
	get_winograd2x2_input_tile_transformation(Bd, in_tiles);


	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){
		if(tx < IMG_DEPTH)
			sImg[tx] = Bd[jj]; 
			
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		__syncthreads();

		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		__syncthreads();
	}
			
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}		
	
}


/* wino3_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		higher than the output channels.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino3_conv2x2_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X];
	__shared__ float sImg[BLOCK_SIZE_X]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	for (int jj = 0; jj < TILE_SIZE; jj++)
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		
	// for(int cnt_out = 0; cnt_out < ceil((double)IMG_DEPTH/(double)BLOCK_SIZE_X); cnt_out++){	
	for(int cnt_out = 0; cnt_out < IMG_DEPTH/BLOCK_SIZE_X; cnt_out++){
		unsigned out_offset = cnt_out*BLOCK_SIZE_X;
		
		// Reset values for padding computations
		#pragma unroll
		for (int cnt = 0; cnt<TILE_SIZE; cnt++)
			in_tiles[cnt] = 0;
	
		// Fetch the input tile data considering the padding
		#pragma unroll
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				#pragma unroll
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
			        	ii = ki + (x_pix - CENTER);

			        	if ((ii >= 0) && (ii <= Nx-1))
				      		in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx + out_offset];
				     
				  }
			  }
		}
		
		float Bd[TILE_SIZE]; 
		get_winograd2x2_input_tile_transformation(Bd, in_tiles);


		// Compute M matrix (M = U.*V) and put values in the shared data array 
 		for (int jj = 0; jj < TILE_SIZE; jj++){
 		
 			sImg[tx] = Bd[jj]; 
			__syncthreads();

			for (int kk = 0; kk < BLOCK_SIZE_X; kk++)
				sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + (out_offset + kk)*out_ch_offset + tx];
				
			__syncthreads();
		}
			
	}		
	
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
	
}




/* wino3b_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		higher than the output channels but the latter is not multiple 
 *		of the former.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino3b_conv2x2_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X];
	__shared__ float sImg[BLOCK_SIZE_X]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	for (int jj = 0; jj < TILE_SIZE; jj++)
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		
	// for(int cnt_out = 0; cnt_out < ceil((double)IMG_DEPTH/(double)BLOCK_SIZE_X); cnt_out++){	
	for(int cnt_out = 0; cnt_out < IMG_DEPTH/BLOCK_SIZE_X + 1; cnt_out++){
		unsigned out_offset = cnt_out*BLOCK_SIZE_X;
		
		if(out_offset + tx < IMG_DEPTH){
			
			// Reset values for padding computations
			#pragma unroll
			for (int cnt = 0; cnt<TILE_SIZE; cnt++)
				in_tiles[cnt] = 0;
		
			// Fetch the input tile data considering the padding
			#pragma unroll
			for (int kj = 0; kj<KERNEL_SIZE; kj++){
				jj = kj + (y_pix - CENTER);
				
				if((jj >= 0) && (jj <= Ny-1)){
					#pragma unroll
					for (int ki = 0; ki<KERNEL_SIZE; ki++){
						ii = ki + (x_pix - CENTER);

						if ((ii >= 0) && (ii <= Nx-1))
					      		in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx + out_offset];
					     
					  }
				  }
			}
		}
		
		// As many Bds as tx
		float Bd[TILE_SIZE]; 
		get_winograd2x2_input_tile_transformation(Bd, in_tiles);


		// Compute M matrix (M = U.*V) and put values in the shared data array 
 		for (int jj = 0; jj < TILE_SIZE; jj++){
 		
 			if(out_offset + tx < IMG_DEPTH)
 				sImg[tx] = Bd[jj];
			else
				sImg[tx] = 0; 
			__syncthreads();

			for (int kk = 0; kk < BLOCK_SIZE_X; kk++)
				sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + (out_offset + kk)*out_ch_offset + tx];
				
			__syncthreads();
		}
			
	}		
	
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
	
}


/* wino4_conv2x2_norm_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the output channels are
 *		lower than the wrap size (i.e., 32).
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is performed.
 *
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
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
__global__ void wino4_conv2x2_norm_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *mean, float *var, float *weight, float *bias, const float EPSILON){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	float mdata[TILE_SIZE]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	for (int jj = 0; jj < TILE_SIZE; jj++)
		mdata[jj] = 0;
		
	for (int kk = 0; kk<Nz; kk++){	
		// Reset values for padding computations
		#pragma unroll
		for (int cnt = 0; cnt<TILE_SIZE; cnt++)
			in_tiles[cnt] = 0;
	
		// Fetch the input tile data considering the padding
		#pragma unroll
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				#pragma unroll
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
					ii = ki + (x_pix - CENTER);

					if ((ii >= 0) && (ii <= Nx-1))
			      			in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + kk];
				     
				  }
			  }
		}
		

		float Bd[TILE_SIZE]; 
		get_winograd2x2_input_tile_transformation(Bd, in_tiles);


		// Compute M matrix (M = U.*V) and put values in the shared data array 
		#pragma unroll
 		for (int jj = 0; jj < TILE_SIZE; jj++)	 		
			mdata[jj] += Bd[jj] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
	
	}	
	

	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, mdata, 1, 0);
			
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Batch normalization 
 			Y[cnt_y*OUT_DIM + cnt_x] = ((Y[cnt_y*OUT_DIM + cnt_x] - *(mean + z_pix))/sqrtf(*(var + z_pix) + EPSILON)) * (*(weight + z_pix)) + *(bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
 			
		  

}





/*******************************************************************************/
/* Functions assuming that batch normalization is included in the filter weghts*/
/*******************************************************************************/



/* wino_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2.
 *		It normalizes the input image, which is defined as an 8 bits   
 *		integer, by 255. The threads move across the output image.
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- uint8_t *img: Non-normilized input image
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino_conv2x2_ReLu_CUDA(float *imgf, uint8_t *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;
	
	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ int sImg[IMG_DEPTH]; 

	int ii, jj;

	int in_tiles[16][IMG_DEPTH];

	// Reset values for padding computations
	#pragma unroll
	for (int cnt = 0; cnt<TILE_SIZE; cnt++){
		in_tiles[cnt][0] = 0;
		in_tiles[cnt][1] = 0;
		in_tiles[cnt][2] = 0;
	}

	// Fetch the input tile data considering the padding
	#pragma unroll
	for (int kj = 0; kj<KERNEL_SIZE; kj++){
		jj = kj + (y_pix - CENTER);
		
		if((jj >= 0) && (jj <= Ny-1)){
			#pragma unroll
			for (int ki = 0; ki<KERNEL_SIZE; ki++){
			      ii = ki + (x_pix - CENTER);

			      if ((ii >= 0) && (ii <= Nx-1)){
				      	in_tiles[kj*KERNEL_SIZE + ki][0] =  (img[jj*Nx*Nz + ii*Nz + 0]);
				      	in_tiles[kj*KERNEL_SIZE + ki][1] =  (img[jj*Nx*Nz + ii*Nz + 1]);
				      	in_tiles[kj*KERNEL_SIZE + ki][2] =  (img[jj*Nx*Nz + ii*Nz + 2]);
			      	}

			  }
		  }
	}

	// As many Bds as tx
	float Bd[TILE_SIZE][IMG_DEPTH];
	#pragma unroll
	for(int cnt = 0; cnt < IMG_DEPTH; cnt++){
		Bd[0][cnt] = (in_tiles[0][cnt]-in_tiles[8][cnt])-(in_tiles[2][cnt]-in_tiles[10][cnt]),       
		Bd[1][cnt] = (in_tiles[1][cnt]-in_tiles[9][cnt])+(in_tiles[2][cnt]-in_tiles[10][cnt]); 
		Bd[2][cnt] = -(in_tiles[1][cnt]-in_tiles[9][cnt])+(in_tiles[2][cnt]-in_tiles[10][cnt]);
		Bd[3][cnt] = (in_tiles[1][cnt]-in_tiles[9][cnt])-(in_tiles[3][cnt]-in_tiles[11][cnt]);
		
		Bd[4][cnt] = (in_tiles[4][cnt]+in_tiles[8][cnt])-(in_tiles[6][cnt]+in_tiles[10][cnt]);       
		Bd[5][cnt] = (in_tiles[5][cnt]+in_tiles[9][cnt])+(in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[6][cnt] = -(in_tiles[5][cnt]+in_tiles[9][cnt])+(in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[7][cnt] = (in_tiles[5][cnt]+in_tiles[9][cnt])-(in_tiles[7][cnt]+in_tiles[11][cnt]);  
	
		Bd[8][cnt] = (-in_tiles[4][cnt]+in_tiles[8][cnt])-(-in_tiles[6][cnt]+in_tiles[10][cnt]);       
		Bd[9][cnt] = (-in_tiles[5][cnt]+in_tiles[9][cnt])+(-in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[10][cnt] = -(-in_tiles[5][cnt]+in_tiles[9][cnt])+(-in_tiles[6][cnt]+in_tiles[10][cnt]); 
		Bd[11][cnt] = (-in_tiles[5][cnt]+in_tiles[9][cnt])-(-in_tiles[7][cnt]+in_tiles[11][cnt]); 
		
		Bd[12][cnt] = (in_tiles[4][cnt]-in_tiles[12][cnt])-(in_tiles[6][cnt]-in_tiles[14][cnt]);       
		Bd[13][cnt] = (in_tiles[5][cnt]-in_tiles[13][cnt])+(in_tiles[6][cnt]-in_tiles[14][cnt]); 
		Bd[14][cnt] = -(in_tiles[5][cnt]-in_tiles[13][cnt])+(in_tiles[6][cnt]-in_tiles[14][cnt]); 
		Bd[15][cnt] = (in_tiles[5][cnt]-in_tiles[13][cnt])-(in_tiles[7][cnt]-in_tiles[15][cnt]); 
	}
	
	

	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){

		sImg[0] = Bd[jj][0];
		sImg[1] = Bd[jj][1];
		sImg[2] = Bd[jj][2];
		
		sdata[jj*out_ch_offset + tx] = 0;
		__syncthreads();
		
		#pragma unroll
		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*out_ch_offset + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		sdata[jj*out_ch_offset + tx] = sdata[jj*out_ch_offset + tx]/255;
		__syncthreads();
	}
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
 			
	
}


/* wino_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		equal to the output channels.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Non-normilized input image
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino_conv2x2_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ float sImg[IMG_DEPTH]; 

	int ii, jj;

	float in_tiles[TILE_SIZE];
	// Reset values for padding computations
	#pragma unroll
	for (int cnt = 0; cnt<TILE_SIZE; cnt++)
		in_tiles[cnt] = 0;

	// Fetch the input tile data considering the padding
	#pragma unroll
	for (int kj = 0; kj<KERNEL_SIZE; kj++){
		jj = kj + (y_pix - CENTER);
		
		if((jj >= 0) && (jj <= Ny-1)){
			#pragma unroll
			for (int ki = 0; ki<KERNEL_SIZE; ki++){
			      ii = ki + (x_pix - CENTER);

			      if ((ii >= 0) && (ii <= Nx-1))
			  	in_tiles[kj*KERNEL_SIZE + ki] = img[jj*Nx*Nz + ii*Nz + tx];

			  }
		  }
	}
	
	// As many Bds as tx
	float Bd[TILE_SIZE]; 
	get_winograd2x2_input_tile_transformation(Bd, in_tiles);
	
	

	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){
	
		sImg[tx] = Bd[jj];
		sdata[jj*out_ch_offset + tx] = 0;
		__syncthreads();

		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*out_ch_offset + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		__syncthreads();
	}
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
 				
}


 /* wino2_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		lower than the output channels.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino2_conv2x2_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ float sImg[IMG_DEPTH]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	// Reset values for padding computations
	#pragma unroll
	for (int cnt = 0; cnt<TILE_SIZE; cnt++)
		in_tiles[cnt] = 0;

	// Fetch the input tile data considering the padding
	#pragma unroll
	for (int kj = 0; kj<KERNEL_SIZE; kj++){
		jj = kj + (y_pix - CENTER);
		
		if((jj >= 0) && (jj <= Ny-1)){
			#pragma unroll
			for (int ki = 0; ki<KERNEL_SIZE; ki++){
			      ii = ki + (x_pix - CENTER);

			      if ((ii >= 0) && (ii <= Nx-1))
			  	in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx];

			     
			  }
		  }
	}
	
	// As many Bds as tx
	float Bd[TILE_SIZE]; 
	get_winograd2x2_input_tile_transformation(Bd, in_tiles);


	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){
	
		sImg[tx] = Bd[jj]; 
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		__syncthreads();

		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		__syncthreads();
	}
			
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}

	
}



/* wino2b_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		lower than the output channels but the former is not multiple 
 *		of the latter.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino2b_conv2x2_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X]; 
	__shared__ float sImg[IMG_DEPTH]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	if(tx < IMG_DEPTH){
		// Reset values for padding computations
		#pragma unroll
		for (int cnt = 0; cnt<TILE_SIZE; cnt++)
			in_tiles[cnt] = 0;

		// Fetch the input tile data considering the padding
		#pragma unroll
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				#pragma unroll
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
				      ii = ki + (x_pix - CENTER);

				      if ((ii >= 0) && (ii <= Nx-1))
				  	in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx];

				     
				  }
			  }
		}
	}
	
	// As many Bds as tx
	float Bd[TILE_SIZE]; 
	get_winograd2x2_input_tile_transformation(Bd, in_tiles);


	// Compute M matrix (M = U.*V) and put values in the shared data array 
	for (int jj = 0; jj < TILE_SIZE; jj++){
		if(tx < IMG_DEPTH)
			sImg[tx] = Bd[jj]; 
			
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		__syncthreads();

		for (int kk = 0; kk < IMG_DEPTH; kk++)
			sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
		__syncthreads();
	}
			
			
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}		
	
}


 /* wino3_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		higher than the output channels.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino3_conv2x2_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X];
	__shared__ float sImg[BLOCK_SIZE_X]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	for (int jj = 0; jj < 16; jj++)
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		
	for(int cnt_out = 0; cnt_out < Nz/BLOCK_SIZE_X; cnt_out++){
		unsigned out_offset = cnt_out*BLOCK_SIZE_X;
		
		// Reset values for padding computations
		#pragma unroll
		for (int cnt = 0; cnt<TILE_SIZE; cnt++)
			in_tiles[cnt] = 0;
	
		// Fetch the input tile data considering the padding
		#pragma unroll
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				#pragma unroll
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
			        	ii = ki + (x_pix - CENTER);

			        	if ((ii >= 0) && (ii <= Nx-1))
				      		in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx + out_offset];
				     
				  }
			  }
		}
		
		// As many Bds as tx
		float Bd[TILE_SIZE]; 
		get_winograd2x2_input_tile_transformation(Bd, in_tiles);


		// Compute M matrix (M = U.*V) and put values in the shared data array 
 		for (int jj = 0; jj < TILE_SIZE; jj++){
 		
 			sImg[tx] = Bd[jj]; 
			__syncthreads();

			for (int kk = 0; kk < BLOCK_SIZE_X; kk++)
				sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + (out_offset + kk)*out_ch_offset + tx];
				
			__syncthreads();
		}
			
	}		
	
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}	  

	
}


/* wino3b_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the input channels are
 *		higher than the output channels but the latter is not multiple 
 *		of the former.
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * template:
 *		- int BLOCK_SIZE_X: Number of threads making up a block 
 *		- int IMG_DEPTH: Depth of the input feature
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE_X, const int IMG_DEPTH> __global__ void wino3b_conv2x2_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;	

	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;
	
	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	// Lock data into the L1 cache
	__shared__ float sdata[TILE_SIZE*BLOCK_SIZE_X];
	__shared__ float sImg[BLOCK_SIZE_X]; 
	float in_tiles[TILE_SIZE];
	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	for (int jj = 0; jj < TILE_SIZE; jj++)
		sdata[jj*BLOCK_SIZE_X + tx] = 0;
		
	// for(int cnt_out = 0; cnt_out < ceil((double)IMG_DEPTH/(double)BLOCK_SIZE_X); cnt_out++){	
	for(int cnt_out = 0; cnt_out < IMG_DEPTH/BLOCK_SIZE_X + 1; cnt_out++){
		unsigned out_offset = cnt_out*BLOCK_SIZE_X;
		
		if(out_offset + tx < IMG_DEPTH){
			
			// Reset values for padding computations
			#pragma unroll
			for (int cnt = 0; cnt<TILE_SIZE; cnt++)
				in_tiles[cnt] = 0;
		
			// Fetch the input tile data considering the padding
			#pragma unroll
			for (int kj = 0; kj<KERNEL_SIZE; kj++){
				jj = kj + (y_pix - CENTER);
				
				if((jj >= 0) && (jj <= Ny-1)){
					#pragma unroll
					for (int ki = 0; ki<KERNEL_SIZE; ki++){
						ii = ki + (x_pix - CENTER);

						if ((ii >= 0) && (ii <= Nx-1))
					      		in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + tx + out_offset];
					     
					  }
				  }
			}
		}
		
		// As many Bds as tx
		float Bd[TILE_SIZE]; 
		get_winograd2x2_input_tile_transformation(Bd, in_tiles);


		// Compute M matrix (M = U.*V) and put values in the shared data array 
 		for (int jj = 0; jj < TILE_SIZE; jj++){
 		
 			if(out_offset + tx < IMG_DEPTH)
 				sImg[tx] = Bd[jj];
			else
				sImg[tx] = 0; 
			__syncthreads();

			for (int kk = 0; kk < BLOCK_SIZE_X; kk++)
				sdata[jj*BLOCK_SIZE_X + tx] += sImg[kk] * kernel[jj*out_ch_offset*Nz + (out_offset + kk)*out_ch_offset + tx];
				
			__syncthreads();
		}
			
	}		
	
				
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, sdata, BLOCK_SIZE_X, tx);
	
	
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}
	
}



 /* wino4_conv2x2_ReLu_CUDA
 *
 * Description: CUDA kernel for performing a Winograd convolution of filter 3x3 
 *		and output matrix of 2x2. To be used when the output channels are
 *		lower than the wrap size (i.e., 32).
 *		The threads move across the output image. Threads synchronize
 *		to load and share the feature data. 
 *		Batch normalization is assumed to be part of the filter weights (integrated).
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input feature
 *		- float *kernel: Matrix with the Winograd kernels (Winograd transformed filters)
 *		- unsigned Nx: img width
 *		- unsigned Ny: img height
 *		- unsigned Nz: img depth
 *		- int out_ch_offset: Number of filters (output channels)	
 *		- float *conv_bias: Convolution bias to add
 *
 * Returns:     Nothing
 *
 * */
__global__ void wino4_conv2x2_ReLu_CUDA(float *imgf, float *img, float *kernel, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float *conv_bias){

	// Block index
  	int bx = blockIdx.x;
  	// Thread index
	int tx = threadIdx.x;
	// Private ID
	unsigned idx = tx + bx*blockDim.x;
	
	// Winograd convolution constants
	const unsigned TILE_SIZE = 16;
	const unsigned OUT_DIM = 2;	
	const unsigned KERNEL_SIZE = 4;
	const unsigned CENTER = 1;	

	// Compute pixel position as function of the private ID
	unsigned temp = idx;
	unsigned z_pix = idx % out_ch_offset;
	temp = (idx / out_ch_offset);
	unsigned x_pix = OUT_DIM*(temp % (Nx/OUT_DIM));
	temp = temp / (Nx/OUT_DIM); 	
	unsigned y_pix = OUT_DIM*temp;

	float mdata[TILE_SIZE]; 
	float in_tiles[TILE_SIZE];

	
	unsigned xz_img_off = Nx*Nz;

	int ii, jj;

	for (int jj = 0; jj < TILE_SIZE; jj++)
		mdata[jj] = 0;
		
	for (int kk = 0; kk<Nz; kk++){	
		// Reset values for padding computations
		#pragma unroll
		for (int cnt = 0; cnt<TILE_SIZE; cnt++)
			in_tiles[cnt] = 0;
	
		// Fetch the input tile data considering the padding
		#pragma unroll
		for (int kj = 0; kj<KERNEL_SIZE; kj++){
			jj = kj + (y_pix - CENTER);
			
			if((jj >= 0) && (jj <= Ny-1)){
				#pragma unroll
				for (int ki = 0; ki<KERNEL_SIZE; ki++){
					ii = ki + (x_pix - CENTER);

					if ((ii >= 0) && (ii <= Nx-1))
			      			in_tiles[kj*KERNEL_SIZE + ki] = img[jj*xz_img_off + ii*Nz + kk];
				     
				  }
			  }
		}
		

		float Bd[TILE_SIZE]; 
		get_winograd2x2_input_tile_transformation(Bd, in_tiles);


		// Compute M matrix (M = U.*V) and put values in the shared data array 
		#pragma unroll
 		for (int jj = 0; jj < TILE_SIZE; jj++)	 		
			mdata[jj] += Bd[jj] * kernel[jj*out_ch_offset*Nz + kk*out_ch_offset + z_pix];
	
	}	
	
	// As many Ys as tx
	float Y[OUT_DIM*OUT_DIM];
	get_winograd2x2_output_tile_transformation(Y, mdata, 1, 0);
			
	// for the 4 output pixels
	#pragma unroll
	for(unsigned cnt_y = 0; cnt_y < OUT_DIM; cnt_y++){
		#pragma unroll
		for(unsigned cnt_x = 0; cnt_x < OUT_DIM; cnt_x++){
			// Add convolution bias
			Y[cnt_y*OUT_DIM + cnt_x] += *(conv_bias + z_pix);

 			// Calculate index
		 	unsigned id_pix = (cnt_y+y_pix)*Nx*out_ch_offset + (cnt_x+x_pix)*out_ch_offset + z_pix;
		
		  	// ReLu (max(0,x))
		 	imgf[id_pix] = (Y[cnt_y*OUT_DIM + cnt_x] < 0) ? 0 : Y[cnt_y*OUT_DIM + cnt_x];

 		}
	}

}



