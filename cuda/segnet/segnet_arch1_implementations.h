// ---------------------------------------------------------------------
// GPU4SEG project
// Copyright (C) 2023 ISAE
// 
// Purpose:
// Evaluation of iGPU for semantic segmentation on embedded systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// alfonso.mascarenas-gonzalez@isae-supaero.fr
// ---------------------------------------------------------------------

/*--------------------------- segnet_arch1_implementations.h ------------------------
|  File segnet_arch1_implementations.h
|
|  Description: Implementations of our 1st Segnet architecture 
|		(original Segnet implementation)
|
|  Version: 1.0
*-----------------------------------------------------------------------------------*/

#ifndef SEGNET_ARCH1_IMPLEMENTATIONS_H_
#define SEGNET_ARCH1_IMPLEMENTATIONS_H_


//#define EVENT_RECORD_TIME 


/* SegnetImplementationOriginalTraditionalArch1
 *
 * Description: Executes the Segnet CNN considering multiple enconder-decoder layers.
 *		 Convolutions are performed using the traditional convolution operation. 	
 *		 Float32 is used. This implementation performs the batch normalization.  
 *
 * Parameter:   
 *		- const dim3 &dimsImg: Dimensions of the resized input image 
 *		- const dim3 &dimsFilter: Dimensions of the filters
 *		- const char INPUT_IMAGE_ADDR[]: Path to input image
 *		- const char BIN_BASE_ADDR[]: Path to binary files directory
 *		- const char OUTPUT_IMAGE_ADDR[]: Path to output image
 *
 * Returns:     Nothing
 *
 * */
void SegnetImplementationOriginalTraditionalArch1(const dim3 &dimsImg, const dim3 &dimsFilter, const char INPUT_IMAGE_ADDR[], const char BIN_BASE_ADDR[], const char OUTPUT_IMAGE_ADDR[]){
 
  #ifdef EVENT_RECORD_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
  #endif
                 
  // Number of encoder-decoder layers 
  int const NB_LAYERS = 5;
  // Horizontal/Vertical downsampling/upsampling value (total size change is given by DOWN_SAMPLING*DOWN_SAMPLING)
  int const DOWN_SAMPLING = 2;
  // Number of classes
  int const NB_CLASSES = 7;
  // Number of channels
  int const NB_CHANNELS = 3;
  // Number of convolutions in this Segent implementation
  const unsigned NB_CONVs = 13;
  // Normalization epsilon value
  const float EPSILON = 0.00001;

  // Convolutions per encoder/decoder layer
  const unsigned NB_CONV_LAYER1 = 2;
  const unsigned NB_CONV_LAYER2 = 2;
  const unsigned NB_CONV_LAYER3 = 3;
  const unsigned NB_CONV_LAYER4 = 3;
  const unsigned NB_CONV_LAYER5 = 3;

  const unsigned CONV_2_LAYER[NB_CONVs] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  const unsigned CONV_PER_LAYER[NB_LAYERS] = {NB_CONV_LAYER1, NB_CONV_LAYER2, NB_CONV_LAYER3, NB_CONV_LAYER4, NB_CONV_LAYER5};
  
  /* **************  Weights, bias and normalization data extraction  ************** */
  char bin_file_path_en_conv[NB_CONVs][128];
  char bin_file_path_de_conv[NB_CONVs][128];
  
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	strcpy(bin_file_path_en_conv[cnt], BIN_BASE_ADDR);
  	strcpy(bin_file_path_de_conv[cnt], BIN_BASE_ADDR);
  }
  
  strcat(bin_file_path_en_conv[0], "11_EncoderConv.bin");
  strcat(bin_file_path_en_conv[1], "12_EncoderConv.bin");
  strcat(bin_file_path_en_conv[2], "21_EncoderConv.bin");
  strcat(bin_file_path_en_conv[3], "22_EncoderConv.bin");
  strcat(bin_file_path_en_conv[4], "31_EncoderConv.bin");
  strcat(bin_file_path_en_conv[5], "32_EncoderConv.bin");
  strcat(bin_file_path_en_conv[6], "33_EncoderConv.bin");
  strcat(bin_file_path_en_conv[7], "41_EncoderConv.bin");
  strcat(bin_file_path_en_conv[8], "42_EncoderConv.bin");
  strcat(bin_file_path_en_conv[9], "43_EncoderConv.bin");
  strcat(bin_file_path_en_conv[10], "51_EncoderConv.bin");
  strcat(bin_file_path_en_conv[11], "52_EncoderConv.bin");
  strcat(bin_file_path_en_conv[12], "53_EncoderConv.bin");  

  strcat(bin_file_path_de_conv[0], "11_DecoderConv.bin");
  strcat(bin_file_path_de_conv[1], "12_DecoderConv.bin");
  strcat(bin_file_path_de_conv[2], "21_DecoderConv.bin");
  strcat(bin_file_path_de_conv[3], "22_DecoderConv.bin");
  strcat(bin_file_path_de_conv[4], "31_DecoderConv.bin");
  strcat(bin_file_path_de_conv[5], "32_DecoderConv.bin");
  strcat(bin_file_path_de_conv[6], "33_DecoderConv.bin");
  strcat(bin_file_path_de_conv[7], "41_DecoderConv.bin");
  strcat(bin_file_path_de_conv[8], "42_DecoderConv.bin");
  strcat(bin_file_path_de_conv[9], "43_DecoderConv.bin");
  strcat(bin_file_path_de_conv[10], "51_DecoderConv.bin");
  strcat(bin_file_path_de_conv[11], "52_DecoderConv.bin");
  strcat(bin_file_path_de_conv[12], "53_DecoderConv.bin"); 
   
   
  struct filter_prop_struct fps_encoder[NB_CONVs], fps_decoder[NB_CONVs];
  
  float *h_filters_encoder[NB_CONVs];
  float *h_filters_decoder[NB_CONVs];
  
  // Extract filters weights and bias
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	load_filters(&h_filters_encoder[cnt], &fps_encoder[cnt], bin_file_path_en_conv[cnt]);
  	load_filters(&h_filters_decoder[cnt], &fps_decoder[cnt], bin_file_path_de_conv[cnt]);
  }
  
  
  char bin_file_path_en_norm[NB_CONVs][128];
  char bin_file_path_de_norm[NB_CONVs][128];
  
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	strcpy(bin_file_path_en_norm[cnt], BIN_BASE_ADDR);
  	strcpy(bin_file_path_de_norm[cnt], BIN_BASE_ADDR);
  }
  
  
  strcat(bin_file_path_en_norm[0], "11_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[1], "12_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[2], "21_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[3], "22_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[4], "31_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[5], "32_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[6], "33_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[7], "41_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[8], "42_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[9], "43_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[10], "51_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[11], "52_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[12], "53_EncoderBatchNorm.bin");  

  strcat(bin_file_path_de_norm[0], "11_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[1], "12_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[2], "21_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[3], "22_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[4], "31_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[5], "32_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[6], "33_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[7], "41_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[8], "42_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[9], "43_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[10], "51_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[11], "52_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[12], "53_DecoderBatchNorm.bin"); 

  
  unsigned nb_norm_features_encoder[NB_CONVs], nb_norm_features_decoder[NB_CONVs];

  float *h_mean_encoder[NB_CONVs];
  float *h_var_encoder[NB_CONVs];
  float *h_weights_norm_encoder[NB_CONVs];
  float *h_bias_norm_encoder[NB_CONVs];
  
  float *h_mean_decoder[NB_CONVs];
  float *h_var_decoder[NB_CONVs];
  float *h_weights_norm_decoder[NB_CONVs];
  float *h_bias_norm_decoder[NB_CONVs];
  

  // Extract mean, var, weight and bias
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	nb_norm_features_encoder[cnt] = load_norm(&h_mean_encoder[cnt], &h_var_encoder[cnt], &h_weights_norm_encoder[cnt], &h_bias_norm_encoder[cnt], bin_file_path_en_norm[cnt]);
  	nb_norm_features_decoder[cnt] = load_norm(&h_mean_decoder[cnt], &h_var_decoder[cnt], &h_weights_norm_decoder[cnt], &h_bias_norm_decoder[cnt], bin_file_path_de_norm[cnt]);
  }

       		
  // Encoder 
  unsigned size_filters_encoder[NB_CONVs];
  unsigned mem_size_filters_encoder[NB_CONVs];
  
  unsigned size_norm_encoder[NB_CONVs];
  unsigned mem_size_norm_encoder[NB_CONVs];
  
  // Decoder
  unsigned size_filters_decoder[NB_CONVs];
  unsigned mem_size_filters_decoder[NB_CONVs];
  
  unsigned size_norm_decoder[NB_CONVs];
  unsigned mem_size_norm_decoder[NB_CONVs];
   
    for (int j = 0; j < NB_CONVs; j++){
    	// Encoder
    	size_filters_encoder[j] = fps_encoder[j].in_channels * fps_encoder[j].out_channels * fps_encoder[j].kernel_size_cols * fps_encoder[j].kernel_size_rows;
    	size_norm_encoder[j] = nb_norm_features_encoder[j];
    	
    	mem_size_filters_encoder[j] = sizeof(float) * size_filters_encoder[j];
    	mem_size_norm_encoder[j] = sizeof(float) * size_norm_encoder[j];
	
    	// Decoder
   	size_filters_decoder[j] = fps_decoder[j].in_channels * fps_decoder[j].out_channels * fps_decoder[j].kernel_size_cols * fps_decoder[j].kernel_size_rows;
    	size_norm_decoder[j] = nb_norm_features_decoder[j];
    	
    	mem_size_filters_decoder[j] = sizeof(float) * size_filters_decoder[j];
    	mem_size_norm_decoder[j] = sizeof(float) * size_norm_decoder[j];
    	
    }
    

  
  /* **************  Features size and memory  ************** */

  // Host feature matrices size and memory 
  unsigned int size_Img_X[NB_LAYERS]; // Image X axis during all layers 
  unsigned int size_Img_Y[NB_LAYERS]; // Image Y axis during all layers 
  unsigned int size_feature[NB_LAYERS]; 
  unsigned int size_pooled[NB_LAYERS]; // Downsampled
  unsigned int size_unpooled[NB_LAYERS]; // Upsampled
    
  unsigned int mem_feature_encoder_conv[NB_CONVs]; 
  unsigned int mem_feature_decoder_conv[NB_CONVs]; 
    
  unsigned int mem_size_pooled[NB_LAYERS]; // Downsampled
  unsigned int mem_size_unpooled[NB_LAYERS]; // Upsampled
  
  unsigned pooled_index[NB_LAYERS] = {1, 3, 6, 9, 12};
  
  size_Img_X[0] = dimsImg.x; // Image X axis
  size_Img_Y[0]  = dimsImg.y; // Image Y axis
  size_feature[0] = dimsImg.x * dimsImg.y; // Feature
  size_pooled[0] =  size_feature[0] / (DOWN_SAMPLING*DOWN_SAMPLING);
  size_unpooled[0] =  size_feature[0];
  
   for (int j = 1; j < NB_LAYERS; j++){
  	size_Img_X[j] = size_Img_X[j-1] / DOWN_SAMPLING; // Image X axis
  	size_Img_Y[j]  = size_Img_Y[j-1] / DOWN_SAMPLING; // Image Y axis
    	size_feature[j] = size_Img_X[j] * size_Img_Y[j]; // Feature  
    	size_pooled[j] = size_feature[j] / (DOWN_SAMPLING*DOWN_SAMPLING); 
    	size_unpooled[j] = size_feature[j]; 
   }
  
   for (int j = 0; j < NB_CONVs; j++){
     	mem_feature_encoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]];
     	mem_feature_decoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]]; 
  }
  
  
   for (int j = 0; j < NB_LAYERS; j++){
   	mem_size_pooled[j] = sizeof(float) * size_pooled[j]; // Downsampled
   	mem_size_unpooled[j] = sizeof(float) * size_unpooled[j]; // Upsampled
   }


  /* **************  Input image  ************** */
    
  // Characteristics of the image to load
  int width_img, height_img, channels_img;
  // Load image
  unsigned char *h_input_img = stbi_load(INPUT_IMAGE_ADDR, &width_img, &height_img, &channels_img, 0); 
	
  if(h_input_img == NULL){
  	fprintf(stderr, "Failed to load the image!\n");
	exit(EXIT_FAILURE);
  }
  	
  // // printf("Loaded image with width %dpx height %dpx and channels %dpx \n", width_img, height_img, channels_img);
  	  
  // Define size and memory space for matrices (before resizing takes place)
  unsigned int size_input_Img = width_img * height_img * channels_img; // Image	
  unsigned int mem_size_input_Img = sizeof(uint8_t) * size_input_Img;	
  
  float x_conv = (float)width_img/(float)dimsImg.x;
  float y_conv = (float)height_img/(float)dimsImg.y;

  /* **************  Image segmentation ************** */
  
  uint8_t *h_output_classes, *h_colored;
  checkCudaErrors(cudaMallocHost(&h_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMallocHost(&h_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));
  
  if (h_output_classes == NULL){
    fprintf(stderr, "Failed to allocate pixels class host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  if (h_colored == NULL){
    fprintf(stderr, "Failed to allocate colored host matrix!\n");
    exit(EXIT_FAILURE);
  }

	
  /* **************  Device Variables Section  ************** */
  
  // Allocate device memory for image convolution
  unsigned int size_resized_input_img = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS;	
  unsigned int mem_size_resized_input_img = sizeof(uint8_t) * size_resized_input_img;
  
  uint8_t *d_input_img, *d_input_img_resized; 
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img), mem_size_input_Img));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img_resized), mem_size_resized_input_img));


  // Filter weights and bias for the device  
  float *d_filters_encoder[NB_CONVs];
  float *d_filters_decoder[NB_CONVs];

  for (int j = 0; j < NB_CONVs; j++){
  	checkCudaErrors(cudaMalloc((&d_filters_encoder[j]), mem_size_filters_encoder[j]));
  	checkCudaErrors(cudaMalloc((&d_filters_decoder[j]), mem_size_filters_decoder[j]));
  }

  // Normalization mean, variance, weights and bias for the device  
  float *d_mean_encoder[NB_CONVs];
  float *d_var_encoder[NB_CONVs];
  float *d_weights_encoder[NB_CONVs];
  float *d_bias_norm_encoder[NB_CONVs];
  
  float *d_mean_decoder[NB_CONVs];
  float *d_var_decoder[NB_CONVs];
  float *d_weights_decoder[NB_CONVs];
  float *d_bias_norm_decoder[NB_CONVs];
  
  
  for (int j = 0; j < NB_CONVs; j++){
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_mean_encoder[j]), mem_size_norm_encoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_var_encoder[j]), mem_size_norm_encoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_weights_encoder[j]), mem_size_norm_encoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_bias_norm_encoder[j]), mem_size_norm_encoder[j]));
  	
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_mean_decoder[j]), mem_size_norm_decoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_var_decoder[j]), mem_size_norm_decoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_weights_decoder[j]), mem_size_norm_decoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_bias_norm_decoder[j]), mem_size_norm_decoder[j]));
  } 
		
  // Allocate device memory for convolutions	
  float *d_features_encoder_l[NB_CONVs];

    for (int j = 0; j < NB_CONVs; j++)
	checkCudaErrors(cudaMalloc((&d_features_encoder_l[j]), fps_encoder[j].out_channels*mem_feature_encoder_conv[j]));
  
  // Allocate device memory for downsampling
  float *d_max_pooled[NB_LAYERS];
  unsigned *d_max_pooled_idx[NB_LAYERS];

   for (int j = 0; j < NB_LAYERS; j++){
	checkCudaErrors(cudaMalloc((&d_max_pooled[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));
	checkCudaErrors(cudaMalloc((&d_max_pooled_idx[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));	
   }
	
  // Allocate device memory for deconvolutions	
  float *d_features_decoder_l[NB_CONVs];
  
  for (int j = 0; j < NB_CONVs; j++)
  	checkCudaErrors(cudaMalloc((&d_features_decoder_l[j]), fps_decoder[j].out_channels*mem_feature_decoder_conv[j]));
   
  // Allocate device memory for upsampling
  float *d_max_unpooled[NB_LAYERS];
   for (int j = 0; j < NB_LAYERS; j++)
	checkCudaErrors(cudaMalloc((&d_max_unpooled[j]), fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]));
	

  // Segmentation memory allocation
  uint8_t *d_output_classes, *d_colored;
  unsigned *d_class_count;
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output_classes), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_colored), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_class_count), sizeof(unsigned)*NB_CLASSES));  



  /* **************  Setup execution parameters  ************** */
 
  // Stream creation	
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));
  
  // Input image resize     
  unsigned threads_per_block_resize = 512; 
  unsigned blocks_per_grid_resize = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS/threads_per_block_resize;
	  
  // First and last convolution  
  unsigned threads_per_block_outer_conv = 512; 
  unsigned blocks_per_grid_outer_conv = ceil((fps_encoder[0].out_channels*size_Img_X[0]*size_Img_Y[0])/threads_per_block_outer_conv);
  
  // The rest of convolutions
  unsigned threads_per_block_conv[NB_CONVs] = {64, 128, 256, 512, 512};
  unsigned blocks_per_grid_conv[NB_CONVs];
  
  for (int i = 0; i < NB_CONVs; i++)
  	blocks_per_grid_conv[i] = ceil((fps_encoder[i].out_channels*size_Img_X[CONV_2_LAYER[i]]*size_Img_Y[CONV_2_LAYER[i]])/threads_per_block_conv[CONV_2_LAYER[i]]); 
       
  // Downsampling operation
  unsigned threads_per_block_pool = 512; 
  unsigned blocks_per_grid_pool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_pool[i] = ceil(((size_Img_X[i]/DOWN_SAMPLING)*(size_Img_Y[i]/DOWN_SAMPLING)*fps_encoder[pooled_index[i]].out_channels)/threads_per_block_pool);
 	
  // Upsampling operation
  unsigned threads_per_block_unpool = 512; 
  unsigned blocks_per_grid_unpool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_unpool[i] = ceil(size_Img_X[i]*size_Img_Y[i]*fps_encoder[pooled_index[i]].out_channels) / threads_per_block_unpool;

  // Select dominant class 
  unsigned threads_per_block_arg_max = 512; 
  unsigned blocks_per_grid_arg_max = ceil((size_Img_X[0]*size_Img_Y[0])/threads_per_block_arg_max);
     
     
  
  /* **************  H2D Transfer  ************** */
	  
  // Input image (not resized)
  checkCudaErrors(cudaMemcpyAsync(d_input_img, h_input_img, mem_size_input_Img, cudaMemcpyHostToDevice, m0_stream));

  // Filters, bias, normalization
  for (int j = 0; j < NB_CONVs; j++){
 	checkCudaErrors(cudaMemcpyAsync(d_filters_encoder[j], h_filters_encoder[j], mem_size_filters_encoder[j], cudaMemcpyHostToDevice, m0_stream)); 
  	checkCudaErrors(cudaMemcpyAsync(d_filters_decoder[j], h_filters_decoder[j], mem_size_filters_decoder[j], cudaMemcpyHostToDevice, m0_stream));
 	
  	checkCudaErrors(cudaMemcpyAsync(d_mean_encoder[j], h_mean_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_var_encoder[j], h_var_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_weights_encoder[j], h_weights_norm_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_bias_norm_encoder[j], h_bias_norm_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	
  	checkCudaErrors(cudaMemcpyAsync(d_mean_decoder[j], h_mean_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_var_decoder[j], h_var_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_weights_decoder[j], h_weights_norm_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_bias_norm_decoder[j], h_bias_norm_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));

  }
  
  
  /* **************  Segnet execution  ************** */
  
  // Allocate CUDA timing events
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  unsigned layer_num = 0;
  unsigned conv_num = 0;

  // The times Segnet is run
  unsigned nIter = 1000; 

  // Execute Segnet compute kernels nIter times
  for (int j = 0; j < nIter; j++){
  
    	#ifdef EVENT_RECORD_CYCLES 
  		// Get GPU time stamp value for the first time
		getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
  	#endif
  	
 	// Record the start event
  	checkCudaErrors(cudaEventRecord(start, m0_stream));
  
   	// Reset arrays
        for (int j = 0; j < NB_LAYERS; j++){
		cudaMemset(&d_max_pooled[j], 0, fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]);
		cudaMemset(&d_max_unpooled[j], 0, fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]);	
   	}
   	
	// Resize input image
	resizeImgCUDA <<<blocks_per_grid_resize, threads_per_block_resize, 0, m0_stream>>> (d_input_img_resized, size_Img_X[0], size_Img_Y[0], d_input_img, width_img, NB_CHANNELS, x_conv, y_conv);

	// Wait for the image to be resized 
	cudaDeviceSynchronize(); 
		
 	 /* **************  Encoder  ************** */	
  	/* **************  Layer 1 - Convolution 1 ************** */	
  	conv_num = 0;
  	layer_num = 0;
	
	// Perform image*filter convolution for the resized image
  	convB_norm_ReLu_CUDA<512, 3> <<<blocks_per_grid_outer_conv, threads_per_block_outer_conv, 0, m0_stream>>>(d_features_encoder_l[conv_num], d_input_img_resized, d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
  		
  	/* **************  Layer 1 - Convolution 2 ************** */	
	conv_num = 1;

	// Perform image*filter convolution 
	convA_norm_ReLu_CUDA<64, 3, 64> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num],  fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num],  d_var_encoder[conv_num],  d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num],  EPSILON);


	// Downsample and save max values indices
	maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);		
			

	/* **************  Layer 2/3/4/5 - Convolution 3-13 ************** */
	for (int cnt_layer = 1; cnt_layer < NB_LAYERS; cnt_layer++){

	  	/* **************  Convolution ************** */
		conv_num++;	
		layer_num++;	 	
	  		
		// Perform convolution after downsampling
		if(layer_num < 2)
			convA_norm_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if(layer_num < 3)
			convA_norm_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if (layer_num >= 3)
			convA_norm_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
					

		/* **************  Convolution  ************** */
			
		// Perform convolution 
		// CONV_PER_LAYER[cnt_layer]-1 because we are already doing one considering the downsampled matrices
		for (int cnt_conv_layer = 0; cnt_conv_layer < CONV_PER_LAYER[cnt_layer]-1; cnt_conv_layer++){
			conv_num++;

		if(layer_num < 2)
			convA_norm_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if(layer_num < 3)
			convA_norm_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if (layer_num >= 3)
			convA_norm_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);

			
		}
	
		// Downsample and save max values indices
		maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);
		
				
	 }
	  		

	  	
	/* **************  Decoder  ************** */
	/* **************  Layer 5/4/3/2/1 - Convolution 13-1 ************** */
	for (int cnt_layer = NB_LAYERS-1; cnt_layer >= 0; cnt_layer--, layer_num--){
	  	/* **************  Upsampling  ************** */	 	
		if(cnt_layer == NB_LAYERS-1){		 		
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_max_pooled[layer_num], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);	
		}else{
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_features_decoder_l[conv_num+1], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);
		}
				
		/* **************  Deconvolution  ************** */	
		// Perform covolution after upsampling
		if(cnt_layer > 2){
			convA_norm_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 1){
			convA_norm_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 0){
			convA_norm_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else{
			convA_norm_ReLu_CUDA<64, 3, 64> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);

		}
		
		// Next convolution
		conv_num--;		

		/* **************  Deconvolution Loop ************** */
		if(cnt_layer > 3){
			convA_norm_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
			
			conv_num--;
			
			convA_norm_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		
		}else if(cnt_layer > 2){
			convA_norm_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
			
			conv_num--;
			
			convA_norm_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num-1], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		
		}else if(cnt_layer > 1){
			convA_norm_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
			
			conv_num--;
			
			convA_norm_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num-1], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 0){
			convA_norm_ReLu_CUDA<64, 3, 64> <<<blocks_per_grid_conv[layer_num-1], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else{
			convB_norm_ReLu_CUDA<512, 3> <<<blocks_per_grid_outer_conv, threads_per_block_outer_conv, 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}

		// Next convolution
		conv_num--;
			
	 }
	 

	/* **************  Segmentation  ************** */		

	// Get the index (class) with the highest value for each pixel
	argMax3D_CUDA<<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_output_classes, d_features_decoder_l[0], NB_CLASSES, size_feature[0]);	
				
	/* **************  Coloring  ************** */
	
	// Array with the color code for each class
	uint8_t d_class_tag_val[NB_CHANNELS];
	
	for(int i = 0; i < NB_CLASSES; i++){
	
		// Background clutter Class 0
		if(i == 0){ 
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;		
		// Building Class 1
		}else if(i == 1){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;
		// Road Class 2
		}else if(i == 2){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 128;
		// Static_Car Class 3
		}else if(i == 3){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 128;
		// Tree Class 4
		}else if (i == 4){
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Vegetation Class 5
		}else if (i == 5){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Human Class 6
		}else if (i == 6){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 0;
		}

		// Color each pixel according to its class
		createRGBclassesCUDA <<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_colored, d_class_count, d_output_classes, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, i, d_class_tag_val[0], d_class_tag_val[1], d_class_tag_val[2]);
		
	}
	
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, m0_stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal;                        
  printf( "Time: %.3f msec \n", msecPerMatrixMul);  
	
  #ifdef EVENT_RECORD_CYCLES 	
	  // Wait for the end of execution of all the threads blocks
	  cudaDeviceSynchronize();   
	  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();   
			    
	  // Calculate execution time in cycles and display it	  
	  calculate_time_diff_clock(1); 
	  print_time_diff_clock();
  #endif	
  }

  // Transfer of the matrix with the colored pixels 
  checkCudaErrors(cudaMemcpy(h_colored, d_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS, cudaMemcpyDeviceToHost)); 
	    	
  // Export colored segmentation image
  stbi_write_png(OUTPUT_IMAGE_ADDR, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, h_colored, size_Img_X[0]*NB_CHANNELS);

  // Transfer of the matrix with the pixel classification 
  checkCudaErrors(cudaMemcpy(h_output_classes, d_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0], cudaMemcpyDeviceToHost)); 

  checkCudaErrors(cudaStreamSynchronize(m0_stream));
  
  
  #ifdef EVENT_RECORD_CYCLES 
   	// Clean up clock memory 
  	clean_clock();
  #endif
  
  // Host
  stbi_image_free(h_input_img);

   for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFreeHost(h_filters_encoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_filters_decoder[cnt]));
   
   	checkCudaErrors(cudaFreeHost(h_mean_encoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_var_encoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_weights_norm_encoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_bias_norm_encoder[cnt]));
   	
   	checkCudaErrors(cudaFreeHost(h_mean_decoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_var_decoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_weights_norm_decoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_bias_norm_decoder[cnt]));
  }
  
  checkCudaErrors(cudaFreeHost(h_output_classes));   
  checkCudaErrors(cudaFreeHost(h_colored));
      
  // Device
  checkCudaErrors(cudaFree(d_input_img));
  checkCudaErrors(cudaFree(d_input_img_resized));
  

  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFree(d_filters_encoder[cnt]));
  	checkCudaErrors(cudaFree(d_filters_decoder[cnt]));
   
   	checkCudaErrors(cudaFree(d_mean_encoder[cnt]));
   	checkCudaErrors(cudaFree(d_var_encoder[cnt]));
   	checkCudaErrors(cudaFree(d_weights_encoder[cnt]));
   	checkCudaErrors(cudaFree(d_bias_norm_encoder[cnt]));
   	
   	checkCudaErrors(cudaFree(d_mean_decoder[cnt]));
   	checkCudaErrors(cudaFree(d_var_decoder[cnt]));
   	checkCudaErrors(cudaFree(d_weights_decoder[cnt]));
   	checkCudaErrors(cudaFree(d_bias_norm_decoder[cnt]));
   	

  	checkCudaErrors(cudaFree(d_features_encoder_l[cnt]));
  	checkCudaErrors(cudaFree(d_features_decoder_l[cnt]));	 	
  }
  
  
  // Upsampling/downsampling
  for(int cnt = 0; cnt < NB_LAYERS; cnt++){
  	
	checkCudaErrors(cudaFree(d_max_pooled[cnt]));
	checkCudaErrors(cudaFree(d_max_pooled_idx[cnt]));
	checkCudaErrors(cudaFree(d_max_unpooled[cnt]));
  }
  
  

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

   
}


/* SegnetImplementationIntegratedTraditionalArch1
 *
 * Description: Executes the Segnet CNN considering multiple enconder-decoder layers.
 *		 Convolutions are performed using the traditional convolution operation. 	
 *		 Float32 is used. The filter weights integrate the batch normalization.   
 *
 * Parameter:   
 *		- const dim3 &dimsImg: Dimensions of the resized input image 
 *		- const dim3 &dimsFilter: Dimensions of the filters
 *		- const char INPUT_IMAGE_ADDR[]: Path to input image
 *		- const char BIN_BASE_ADDR[]: Path to binary files directory
 *		- const char OUTPUT_IMAGE_ADDR[]: Path to output image
 *
 * Returns:     Nothing
 *
 * */
void SegnetImplementationIntegratedTraditionalArch1(const dim3 &dimsImg, const dim3 &dimsFilter, const char INPUT_IMAGE_ADDR[], const char BIN_BASE_ADDR[], const char OUTPUT_IMAGE_ADDR[]){
 
  #ifdef EVENT_RECORD_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
  #endif
                 
  // Number of encoder-decoder layers 
  int const NB_LAYERS = 5;
  // Horizontal/Vertical downsampling/upsampling value (total size change is given by DOWN_SAMPLING*DOWN_SAMPLING)
  int const DOWN_SAMPLING = 2;
  // Number of classes
  int const NB_CLASSES = 7;
  // Number of channels
  int const NB_CHANNELS = 3;
  // Number of convolutions in this Segent implementation
  const unsigned NB_CONVs = 13;

  // Convolutions per encoder/decoder layer
  const unsigned NB_CONV_LAYER1 = 2;
  const unsigned NB_CONV_LAYER2 = 2;
  const unsigned NB_CONV_LAYER3 = 3;
  const unsigned NB_CONV_LAYER4 = 3;
  const unsigned NB_CONV_LAYER5 = 3;

  const unsigned CONV_2_LAYER[13] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  const unsigned CONV_PER_LAYER[5] = {NB_CONV_LAYER1, NB_CONV_LAYER2, NB_CONV_LAYER3, NB_CONV_LAYER4, NB_CONV_LAYER5};
  
  /* **************  Weights, bias and normalization data extraction  ************** */
  char bin_file_path_en_conv[NB_CONVs][128];
  char bin_file_path_de_conv[NB_CONVs][128];
  
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	strcpy(bin_file_path_en_conv[cnt], BIN_BASE_ADDR);
  	strcpy(bin_file_path_de_conv[cnt], BIN_BASE_ADDR);
  }
  
  strcat(bin_file_path_en_conv[0], "11_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[1], "12_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[2], "21_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[3], "22_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[4], "31_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[5], "32_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[6], "33_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[7], "41_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[8], "42_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[9], "43_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[10], "51_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[11], "52_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[12], "53_EncoderIntegratedConvAndBatchNorm.bin");  

  strcat(bin_file_path_de_conv[0], "11_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[1], "12_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[2], "21_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[3], "22_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[4], "31_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[5], "32_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[6], "33_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[7], "41_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[8], "42_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[9], "43_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[10], "51_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[11], "52_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[12], "53_DecoderIntegratedConvAndBatchNorm.bin"); 
   
   
  struct filter_prop_struct fps_encoder[NB_CONVs], fps_decoder[NB_CONVs];
  
  float *h_filters_encoder[NB_CONVs];
  float *h_bias_encoder[NB_CONVs];
  
  float *h_filters_decoder[NB_CONVs];
  float *h_bias_decoder[NB_CONVs];
  
  // Extract filters weights and bias
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	load_filters_integrated(&h_filters_encoder[cnt], &h_bias_encoder[cnt], &fps_encoder[cnt], bin_file_path_en_conv[cnt]);
  	load_filters_integrated(&h_filters_decoder[cnt], &h_bias_decoder[cnt], &fps_decoder[cnt], bin_file_path_de_conv[cnt]);
  }
		
  // Encoder 
  unsigned size_filters_encoder[NB_CONVs];
  unsigned size_bias_encoder[NB_CONVs];
    
  unsigned mem_size_filters_encoder[NB_CONVs];
  unsigned mem_size_bias_encoder[NB_CONVs];
  
  // Decoder
  unsigned size_filters_decoder[NB_CONVs];
  unsigned size_bias_decoder[NB_CONVs];
    
  unsigned mem_size_filters_decoder[NB_CONVs];
  unsigned mem_size_bias_decoder[NB_CONVs];

   
    for (int j = 0; j < NB_CONVs; j++){
    	// Encoder
    	size_filters_encoder[j] = fps_encoder[j].in_channels * fps_encoder[j].out_channels * fps_encoder[j].kernel_size_cols * fps_encoder[j].kernel_size_rows;
    	size_bias_encoder[j] = fps_encoder[j].out_channels;
    	
    	mem_size_filters_encoder[j] = sizeof(float) * size_filters_encoder[j];
    	mem_size_bias_encoder[j] = sizeof(float) * size_bias_encoder[j];
    		
    	// Decoder
   	size_filters_decoder[j] = fps_decoder[j].in_channels * fps_decoder[j].out_channels * fps_decoder[j].kernel_size_cols * fps_decoder[j].kernel_size_rows;
    	size_bias_decoder[j] = fps_decoder[j].out_channels;
    	
    	mem_size_filters_decoder[j] = sizeof(float) * size_filters_decoder[j];
    	mem_size_bias_decoder[j] = sizeof(float) * size_bias_decoder[j];

    }
    

  
  /* **************  Features size and memory  ************** */

  // Host feature matrices size and memory 
  unsigned int size_Img_X[NB_LAYERS]; // Image X axis during all layers 
  unsigned int size_Img_Y[NB_LAYERS]; // Image Y axis during all layers 
  unsigned int size_feature[NB_LAYERS]; 
  unsigned int size_pooled[NB_LAYERS]; // Downsampled
  unsigned int size_unpooled[NB_LAYERS]; // Upsampled
    
  unsigned int mem_feature_encoder_conv[NB_CONVs]; 
  unsigned int mem_feature_decoder_conv[NB_CONVs]; 
    
  unsigned int mem_size_pooled[NB_LAYERS]; // Downsampled
  unsigned int mem_size_unpooled[NB_LAYERS]; // Upsampled
  
  unsigned pooled_index[NB_LAYERS] = {1, 3, 6, 9, 12};
  
  size_Img_X[0] = dimsImg.x; // Image X axis
  size_Img_Y[0]  = dimsImg.y; // Image Y axis
  size_feature[0] = dimsImg.x * dimsImg.y; // Feature
  size_pooled[0] =  size_feature[0] / (DOWN_SAMPLING*DOWN_SAMPLING);
  size_unpooled[0] =  size_feature[0];
  
   for (int j = 1; j < NB_LAYERS; j++){
  	size_Img_X[j] = size_Img_X[j-1] / DOWN_SAMPLING; // Image X axis
  	size_Img_Y[j]  = size_Img_Y[j-1] / DOWN_SAMPLING; // Image Y axis
    	size_feature[j] = size_Img_X[j] * size_Img_Y[j]; // Feature  
    	size_pooled[j] = size_feature[j] / (DOWN_SAMPLING*DOWN_SAMPLING); 
    	size_unpooled[j] = size_feature[j]; 
   }
  
   for (int j = 0; j < NB_CONVs; j++){
     	mem_feature_encoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]];
     	mem_feature_decoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]]; 
  }
  
  
   for (int j = 0; j < NB_LAYERS; j++){
   	mem_size_pooled[j] = sizeof(float) * size_pooled[j]; // Downsampled
   	mem_size_unpooled[j] = sizeof(float) * size_unpooled[j]; // Upsampled
   }


  /* **************  Input image  ************** */
    
  // Characteristics of the image to load
  int width_img, height_img, channels_img;
  // Load image
  unsigned char *h_input_img = stbi_load(INPUT_IMAGE_ADDR, &width_img, &height_img, &channels_img, 0); 
	
  if(h_input_img == NULL){
  	fprintf(stderr, "Failed to load the image!\n");
	exit(EXIT_FAILURE);
  }
  	
  // // printf("Loaded image with width %dpx height %dpx and channels %dpx \n", width_img, height_img, channels_img);
  	  
  // Define size and memory space for matrices (before resizing takes place)
  unsigned int size_input_Img = width_img * height_img * channels_img; // Image	
  unsigned int mem_size_input_Img = sizeof(uint8_t) * size_input_Img;	
  
  float x_conv = (float)width_img/(float)dimsImg.x;
  float y_conv = (float)height_img/(float)dimsImg.y;

  /* **************  Image segmentation ************** */
  
  uint8_t *h_output_classes, *h_colored;
  checkCudaErrors(cudaMallocHost(&h_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMallocHost(&h_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));
  
  if (h_output_classes == NULL){
    fprintf(stderr, "Failed to allocate pixels class host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  if (h_colored == NULL){
    fprintf(stderr, "Failed to allocate colored host matrix!\n");
    exit(EXIT_FAILURE);
  }

	
  /* **************  Device Variables Section  ************** */
  
  // Allocate device memory for image convolution
  unsigned int size_resized_input_img = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS;	
  unsigned int mem_size_resized_input_img = sizeof(uint8_t) * size_resized_input_img;
  
  uint8_t *d_input_img, *d_input_img_resized; 
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img), mem_size_input_Img));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img_resized), mem_size_resized_input_img));


  // Filter weights and bias for the device  
  float *d_bias_encoder[NB_CONVs];
  float *d_bias_decoder[NB_CONVs]; 
  float *d_filters_encoder[NB_CONVs];
  float *d_filters_decoder[NB_CONVs];

  for (int j = 0; j < NB_CONVs; j++){
    	checkCudaErrors(cudaMalloc((&d_bias_encoder[j]), mem_size_bias_encoder[j]));
  	checkCudaErrors(cudaMalloc((&d_bias_decoder[j]), mem_size_bias_decoder[j]));
  	
  	checkCudaErrors(cudaMalloc((&d_filters_encoder[j]), mem_size_filters_encoder[j]));
  	checkCudaErrors(cudaMalloc((&d_filters_decoder[j]), mem_size_filters_decoder[j]));
  }
		
  // Allocate device memory for convolutions	
  float *d_features_encoder_l[NB_CONVs];

  for (int j = 0; j < NB_CONVs; j++)
	checkCudaErrors(cudaMalloc((&d_features_encoder_l[j]), fps_encoder[j].out_channels*mem_feature_encoder_conv[j]));
  
  // Allocate device memory for downsampling
  float *d_max_pooled[NB_LAYERS];
  unsigned *d_max_pooled_idx[NB_LAYERS];

  for (int j = 0; j < NB_LAYERS; j++){
	checkCudaErrors(cudaMalloc((&d_max_pooled[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));
	checkCudaErrors(cudaMalloc((&d_max_pooled_idx[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));	
   }
	
  // Allocate device memory for deconvolutions	
  float *d_features_decoder_l[NB_CONVs];
  
  for (int j = 0; j < NB_CONVs; j++)
  	checkCudaErrors(cudaMalloc((&d_features_decoder_l[j]), fps_decoder[j].out_channels*mem_feature_decoder_conv[j]));
   
  // Allocate device memory for upsampling
  float *d_max_unpooled[NB_LAYERS];
  for (int j = 0; j < NB_LAYERS; j++)
	checkCudaErrors(cudaMalloc((&d_max_unpooled[j]), fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]));
	

  // Segmentation memory allocation
  uint8_t *d_output_classes, *d_colored;
  unsigned *d_class_count;
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output_classes), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_colored), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_class_count), sizeof(unsigned)*NB_CLASSES));  



  /* **************  Setup execution parameters  ************** */
 
  // Stream creation	
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));
  
  // Input image resize     
  unsigned threads_per_block_resize = 512; 
  unsigned blocks_per_grid_resize = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS/threads_per_block_resize;
	  
  // First and last convolution  
  unsigned threads_per_block_outer_conv = 512; 
  unsigned blocks_per_grid_outer_conv = ceil((fps_encoder[0].out_channels*size_Img_X[0]*size_Img_Y[0])/threads_per_block_outer_conv);
  
  // The rest of convolutions
  unsigned threads_per_block_conv[NB_CONVs] = {64, 128, 256, 512, 512};
  unsigned blocks_per_grid_conv[NB_CONVs];
  
  for (int i = 0; i < NB_CONVs; i++)
  	blocks_per_grid_conv[i] = ceil((fps_encoder[i].out_channels*size_Img_X[CONV_2_LAYER[i]]*size_Img_Y[CONV_2_LAYER[i]])/threads_per_block_conv[CONV_2_LAYER[i]]); 
       
  // Downsampling operation
  unsigned threads_per_block_pool = 512; 
  unsigned blocks_per_grid_pool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_pool[i] = ceil(((size_Img_X[i]/DOWN_SAMPLING)*(size_Img_Y[i]/DOWN_SAMPLING)*fps_encoder[pooled_index[i]].out_channels)/threads_per_block_pool);
 	
  // Upsampling operation
  unsigned threads_per_block_unpool = 512; 
  unsigned blocks_per_grid_unpool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_unpool[i] = ceil(size_Img_X[i]*size_Img_Y[i]*fps_encoder[pooled_index[i]].out_channels) / threads_per_block_unpool;

  // Select dominant class 
  unsigned threads_per_block_arg_max = 512; 
  unsigned blocks_per_grid_arg_max = ceil((size_Img_X[0]*size_Img_Y[0])/threads_per_block_arg_max);
     
     
  
  /* **************  H2D Transfer  ************** */
	  
  // Input image (not resized)
  checkCudaErrors(cudaMemcpyAsync(d_input_img, h_input_img, mem_size_input_Img, cudaMemcpyHostToDevice, m0_stream));

  // Filters, bias
  for (int j = 0; j < NB_CONVs; j++){
    	checkCudaErrors(cudaMemcpyAsync(d_bias_encoder[j], h_bias_encoder[j], mem_size_bias_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_bias_decoder[j], h_bias_decoder[j], mem_size_bias_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	
 	checkCudaErrors(cudaMemcpyAsync(d_filters_encoder[j], h_filters_encoder[j], mem_size_filters_encoder[j], cudaMemcpyHostToDevice, m0_stream)); 
  	checkCudaErrors(cudaMemcpyAsync(d_filters_decoder[j], h_filters_decoder[j], mem_size_filters_decoder[j], cudaMemcpyHostToDevice, m0_stream));

  }
  
  
  /* **************  Segnet execution  ************** */
  
  // Allocate CUDA timing events
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  unsigned layer_num = 0;
  unsigned conv_num = 0;

  // The times Segnet is run
  unsigned nIter = 1000; 

  // Execute Segnet compute kernels nIter times
  for (int j = 0; j < nIter; j++){
  
    	#ifdef EVENT_RECORD_CYCLES 
  		// Get GPU time stamp value for the first time
		getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
  	#endif
  	
 	// Record the start event
  	checkCudaErrors(cudaEventRecord(start, m0_stream));
  
   	// Reset arrays
        for (int j = 0; j < NB_LAYERS; j++){
		cudaMemset(&d_max_pooled[j], 0, fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]);
		cudaMemset(&d_max_unpooled[j], 0, fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]);	
   	}
   	
	// Resize input image
	resizeImgCUDA <<<blocks_per_grid_resize, threads_per_block_resize, 0, m0_stream>>> (d_input_img_resized, size_Img_X[0], size_Img_Y[0], d_input_img, width_img, NB_CHANNELS, x_conv, y_conv);

	// Wait for the image to be resized 
	cudaDeviceSynchronize(); 
		
 	 /* **************  Encoder  ************** */	
  	/* **************  Layer 1 - Convolution 1 ************** */	
  	conv_num = 0;
  	layer_num = 0;
	
	// Perform image*filter convolution for the resized image
  	convB_ReLu_CUDA<512, 3> <<<blocks_per_grid_outer_conv, threads_per_block_outer_conv, 0, m0_stream>>>(d_features_encoder_l[conv_num], d_input_img_resized, d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
  		
  	/* **************  Layer 1 - Convolution 2 ************** */	
	conv_num = 1;

	// Perform image*filter convolution 
	convA_ReLu_CUDA<64, 3, 64> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num],  fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);


	// Downsample and save max values indices
	maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);		
			

	/* **************  Layer 2/3/4/5 - Convolution 3-13 ************** */
	for (int cnt_layer = 1; cnt_layer < NB_LAYERS; cnt_layer++){

	  	/* **************  Convolution ************** */
		conv_num++;	
		layer_num++;	 	
	  		
		// Perform convolution after downsampling
		if(layer_num < 2)
			convA_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
		else if(layer_num < 3)
			convA_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
		else if (layer_num >= 3)
			convA_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels,  fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
					

		/* **************  Convolution  ************** */
			
		// Perform convolution 
		// CONV_PER_LAYER[cnt_layer]-1 because we are already doing one considering the downsampled matrices
		for (int cnt_conv_layer = 0; cnt_conv_layer < CONV_PER_LAYER[cnt_layer]-1; cnt_conv_layer++){
			conv_num++;

		if(layer_num < 2)
			convA_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
		else if(layer_num < 3)
			convA_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
		else if (layer_num >= 3)
			convA_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
			
		}
	
		// Downsample and save max values indices
		maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);
		
				
	 }
	  		

	  	
	/* **************  Decoder  ************** */
	/* **************  Layer 5/4/3/2/1 - Convolution 13-1 ************** */
	for (int cnt_layer = NB_LAYERS-1; cnt_layer >= 0; cnt_layer--, layer_num--){
	  	/* **************  Upsampling  ************** */	 	
		if(cnt_layer == NB_LAYERS-1){		 		
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_max_pooled[layer_num], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);	
		}else{
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_features_decoder_l[conv_num+1], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);
		}
				
		/* **************  Deconvolution  ************** */	
		// Perform covolution after upsampling
		if(cnt_layer > 2){
			convA_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else if(cnt_layer > 1){
			convA_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else if(cnt_layer > 0){
			convA_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else{
			convA_ReLu_CUDA<64, 3, 64> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);

		}
		
		// Next convolution
		conv_num--;		

		/* **************  Deconvolution Loop ************** */
		if(cnt_layer > 3){
			convA_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
			
			conv_num--;
			
			convA_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		
		}else if(cnt_layer > 2){
			convA_ReLu_CUDA<512, 3, 512> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
			
			conv_num--;
			
			convA_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num-1], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		
		}else if(cnt_layer > 1){
			convA_ReLu_CUDA<256, 3, 256> <<<blocks_per_grid_conv[layer_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
			
			conv_num--;
			
			convA_ReLu_CUDA<128, 3, 128> <<<blocks_per_grid_conv[layer_num-1], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else if(cnt_layer > 0){
			convA_ReLu_CUDA<64, 3, 64> <<<blocks_per_grid_conv[layer_num-1], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else{
			convB_ReLu_CUDA<512, 3> <<<blocks_per_grid_outer_conv, threads_per_block_outer_conv, 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}

		// Next convolution
		conv_num--;
			
	 }
	 

	/* **************  Segmentation  ************** */		

	// Get the index (class) with the highest value for each pixel
	argMax3D_CUDA<<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_output_classes, d_features_decoder_l[0], NB_CLASSES, size_feature[0]);	
				
	/* **************  Coloring  ************** */
	
	// Array with the color code for each class
	uint8_t d_class_tag_val[NB_CHANNELS];
	
	for(int i = 0; i < NB_CLASSES; i++){
	
		// Background clutter Class 0
		if(i == 0){ 
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;		
		// Building Class 1
		}else if(i == 1){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;
		// Road Class 2
		}else if(i == 2){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 128;
		// Static_Car Class 3
		}else if(i == 3){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 128;
		// Tree Class 4
		}else if (i == 4){
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Vegetation Class 5
		}else if (i == 5){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Human Class 6
		}else if (i == 6){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 0;
		}

		// Color each pixel according to its class
		createRGBclassesCUDA <<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_colored, d_class_count, d_output_classes, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, i, d_class_tag_val[0], d_class_tag_val[1], d_class_tag_val[2]);
		
	}
	
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, m0_stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal;                        
  printf( "Time: %.3f msec \n", msecPerMatrixMul);  
		
  #ifdef EVENT_RECORD_CYCLES 	
	  // Wait for the end of execution of all the threads blocks
	  cudaDeviceSynchronize();   
	  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();   
			    
	  // Calculate execution time in cycles and display it	  
	  calculate_time_diff_clock(1); 
	  print_time_diff_clock();
  #endif
  }

  // Transfer of the matrix with the colored pixels 
  checkCudaErrors(cudaMemcpy(h_colored, d_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS, cudaMemcpyDeviceToHost)); 
	    	
  // Export colored segmentation image
  stbi_write_png(OUTPUT_IMAGE_ADDR, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, h_colored, size_Img_X[0]*NB_CHANNELS);

  // Transfer of the matrix with the pixel classification 
  checkCudaErrors(cudaMemcpy(h_output_classes, d_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0], cudaMemcpyDeviceToHost)); 

  checkCudaErrors(cudaStreamSynchronize(m0_stream));
  
  
  #ifdef EVENT_RECORD_CYCLES 
   	// Clean up clock memory 
  	clean_clock();
  #endif
  
  // Host
  stbi_image_free(h_input_img);

   for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFreeHost(h_filters_encoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_filters_decoder[cnt]));
   
  	checkCudaErrors(cudaFreeHost(h_bias_encoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_bias_decoder[cnt]));
  }
  
  checkCudaErrors(cudaFreeHost(h_output_classes));   
  checkCudaErrors(cudaFreeHost(h_colored));
      
  // Device
  checkCudaErrors(cudaFree(d_input_img));
  checkCudaErrors(cudaFree(d_input_img_resized));
  

  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFree(d_filters_encoder[cnt]));
  	checkCudaErrors(cudaFree(d_filters_decoder[cnt]));
   
  	checkCudaErrors(cudaFree(d_bias_encoder[cnt]));
  	checkCudaErrors(cudaFree(d_bias_decoder[cnt]));
   	
  	checkCudaErrors(cudaFree(d_features_encoder_l[cnt]));
  	checkCudaErrors(cudaFree(d_features_decoder_l[cnt]));	 	
  }
  
  
  // Upsampling/downsampling
  for(int cnt = 0; cnt < NB_LAYERS; cnt++){
  	
	checkCudaErrors(cudaFree(d_max_pooled[cnt]));
	checkCudaErrors(cudaFree(d_max_pooled_idx[cnt]));
	checkCudaErrors(cudaFree(d_max_unpooled[cnt]));
  }
  
  

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

   
}


/* SegnetImplementationOriginalWinograd2x2Arch1
 *
 * Description: Executes the Segnet CNN considering multiple enconder-decoder layers.
 *		 Convolutions are made via Winograd. This implementation performs 
 *		 the batch normalization. Float32 is used.	
 *
 * Parameter:   
 *		- const dim3 &dimsImg: Dimensions of the resized input image 
 *		- const dim3 &dimsFilter: Dimensions of the filters
 *		- const char INPUT_IMAGE_ADDR[]: Path to input image
 *		- const char BIN_BASE_ADDR[]: Path to binary files directory
 *		- const char OUTPUT_IMAGE_ADDR[]: Path to output image
 *
 * Returns:     Nothing
 *
 * */
void SegnetImplementationOriginalWinograd2x2Arch1(const dim3 &dimsImg, const dim3 &dimsFilter, const char INPUT_IMAGE_ADDR[], const char BIN_BASE_ADDR[], const char OUTPUT_IMAGE_ADDR[]){
 
  #ifdef EVENT_RECORD_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
  #endif
                 
  // Number of encoder-decoder layers
  int const NB_LAYERS = 5;
  // Horizontal/Vertical downsampling/upsampling value (total size change is given by DOWN_SAMPLING*DOWN_SAMPLING)
  int const DOWN_SAMPLING = 2;
  // Number of classes
  int const NB_CLASSES = 7;
  // Number of channels
  int const NB_CHANNELS = 3;
  // Number of convolutions in this Segent implementation
  const unsigned NB_CONVs = 13;
  // Normalization epsilon value
  const float EPSILON = 0.00001;
  // Output Winograd convolution tile dimension
  const unsigned WINO_TILE_SIZE = 4;

  // Convolutions per encoder/decoder layer
  const unsigned NB_CONV_LAYER1 = 2;
  const unsigned NB_CONV_LAYER2 = 2;
  const unsigned NB_CONV_LAYER3 = 3;
  const unsigned NB_CONV_LAYER4 = 3;
  const unsigned NB_CONV_LAYER5 = 3;

  const unsigned CONV_2_LAYER[NB_CONVs] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  const unsigned CONV_PER_LAYER[NB_LAYERS] = {NB_CONV_LAYER1, NB_CONV_LAYER2, NB_CONV_LAYER3, NB_CONV_LAYER4, NB_CONV_LAYER5};

  /* **************  Weights, bias and normalization data extraction  ************** */
  
  char bin_file_path_en_conv[NB_CONVs][128];
  char bin_file_path_de_conv[NB_CONVs][128];
  
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	strcpy(bin_file_path_en_conv[cnt], BIN_BASE_ADDR);
  	strcpy(bin_file_path_de_conv[cnt], BIN_BASE_ADDR);
  }
  
  strcat(bin_file_path_en_conv[0], "11_EncoderConv.bin");
  strcat(bin_file_path_en_conv[1], "12_EncoderConv.bin");
  strcat(bin_file_path_en_conv[2], "21_EncoderConv.bin");
  strcat(bin_file_path_en_conv[3], "22_EncoderConv.bin");
  strcat(bin_file_path_en_conv[4], "31_EncoderConv.bin");
  strcat(bin_file_path_en_conv[5], "32_EncoderConv.bin");
  strcat(bin_file_path_en_conv[6], "33_EncoderConv.bin");
  strcat(bin_file_path_en_conv[7], "41_EncoderConv.bin");
  strcat(bin_file_path_en_conv[8], "42_EncoderConv.bin");
  strcat(bin_file_path_en_conv[9], "43_EncoderConv.bin");
  strcat(bin_file_path_en_conv[10], "51_EncoderConv.bin");
  strcat(bin_file_path_en_conv[11], "52_EncoderConv.bin");
  strcat(bin_file_path_en_conv[12], "53_EncoderConv.bin");  

  strcat(bin_file_path_de_conv[0], "11_DecoderConv.bin");
  strcat(bin_file_path_de_conv[1], "12_DecoderConv.bin");
  strcat(bin_file_path_de_conv[2], "21_DecoderConv.bin");
  strcat(bin_file_path_de_conv[3], "22_DecoderConv.bin");
  strcat(bin_file_path_de_conv[4], "31_DecoderConv.bin");
  strcat(bin_file_path_de_conv[5], "32_DecoderConv.bin");
  strcat(bin_file_path_de_conv[6], "33_DecoderConv.bin");
  strcat(bin_file_path_de_conv[7], "41_DecoderConv.bin");
  strcat(bin_file_path_de_conv[8], "42_DecoderConv.bin");
  strcat(bin_file_path_de_conv[9], "43_DecoderConv.bin");
  strcat(bin_file_path_de_conv[10], "51_DecoderConv.bin");
  strcat(bin_file_path_de_conv[11], "52_DecoderConv.bin");
  strcat(bin_file_path_de_conv[12], "53_DecoderConv.bin"); 
   
   
  struct filter_prop_struct fps_encoder[NB_CONVs], fps_decoder[NB_CONVs];
  
  float *h_filters_encoder[NB_CONVs];
  float *h_filters_decoder[NB_CONVs];
  
  // Extract filters weights and bias
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	load_filters(&h_filters_encoder[cnt], &fps_encoder[cnt], bin_file_path_en_conv[cnt]);
  	load_filters(&h_filters_decoder[cnt], &fps_decoder[cnt], bin_file_path_de_conv[cnt]);
  }
  

  float *h_wino_filters_encoder[NB_CONVs];
  float *h_wino_filters_decoder[NB_CONVs];
  

  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	wino_3x3filter_2x2transform(&h_wino_filters_encoder[cnt], h_filters_encoder[cnt], fps_encoder[cnt]);
  	wino_3x3filter_2x2transform(&h_wino_filters_decoder[cnt], h_filters_decoder[cnt], fps_decoder[cnt]);
  }


  char bin_file_path_en_norm[NB_CONVs][128];
  char bin_file_path_de_norm[NB_CONVs][128];
  
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	strcpy(bin_file_path_en_norm[cnt], BIN_BASE_ADDR);
  	strcpy(bin_file_path_de_norm[cnt], BIN_BASE_ADDR);
  }

  strcat(bin_file_path_en_norm[0], "11_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[1], "12_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[2], "21_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[3], "22_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[4], "31_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[5], "32_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[6], "33_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[7], "41_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[8], "42_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[9], "43_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[10], "51_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[11], "52_EncoderBatchNorm.bin");
  strcat(bin_file_path_en_norm[12], "53_EncoderBatchNorm.bin");  

  strcat(bin_file_path_de_norm[0], "11_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[1], "12_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[2], "21_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[3], "22_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[4], "31_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[5], "32_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[6], "33_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[7], "41_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[8], "42_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[9], "43_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[10], "51_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[11], "52_DecoderBatchNorm.bin");
  strcat(bin_file_path_de_norm[12], "53_DecoderBatchNorm.bin"); 
  
  
  unsigned nb_norm_features_encoder[NB_CONVs], nb_norm_features_decoder[NB_CONVs];

  float *h_mean_encoder[NB_CONVs];
  float *h_var_encoder[NB_CONVs];
  float *h_weights_norm_encoder[NB_CONVs];
  float *h_bias_norm_encoder[NB_CONVs];
  
  float *h_mean_decoder[NB_CONVs];
  float *h_var_decoder[NB_CONVs];
  float *h_weights_norm_decoder[NB_CONVs];
  float *h_bias_norm_decoder[NB_CONVs];
  
  
  // Extract mean, var, weight and bias
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	nb_norm_features_encoder[cnt] = load_norm(&h_mean_encoder[cnt], &h_var_encoder[cnt], &h_weights_norm_encoder[cnt], &h_bias_norm_encoder[cnt], bin_file_path_en_norm[cnt]);
  	nb_norm_features_decoder[cnt] = load_norm(&h_mean_decoder[cnt], &h_var_decoder[cnt], &h_weights_norm_decoder[cnt], &h_bias_norm_decoder[cnt], bin_file_path_de_norm[cnt]);
  }

       		
  // Encoder 
  unsigned size_wino_filters_encoder[NB_CONVs];
  unsigned mem_size_wino_filters_encoder[NB_CONVs];
  
  unsigned size_norm_encoder[NB_CONVs];
  unsigned mem_size_norm_encoder[NB_CONVs];

  // Decoder
  unsigned size_wino_filters_decoder[NB_CONVs];
  unsigned mem_size_wino_filters_decoder[NB_CONVs];
  
  unsigned size_norm_decoder[NB_CONVs];
  unsigned mem_size_norm_decoder[NB_CONVs];
   
    for (int j = 0; j < NB_CONVs; j++){
    	// Encoder
    	size_wino_filters_encoder[j] = fps_encoder[j].in_channels * fps_encoder[j].out_channels * 16; // for 3x3 filters and 2x2 output convolution tile
    	size_norm_encoder[j] = nb_norm_features_encoder[j];
    	
    	mem_size_wino_filters_encoder[j] = sizeof(float) * size_wino_filters_encoder[j];
    	mem_size_norm_encoder[j] = sizeof(float) * size_norm_encoder[j];
    		
    	// Decoder
    	size_wino_filters_decoder[j] = fps_decoder[j].in_channels * fps_decoder[j].out_channels * 16; // for 3x3 filters and 2x2 output convolution tile
    	size_norm_decoder[j] = nb_norm_features_decoder[j];
    	
    	mem_size_wino_filters_decoder[j] = sizeof(float) * size_wino_filters_decoder[j];
    	mem_size_norm_decoder[j] = sizeof(float) * size_norm_decoder[j];
    }
    

  
  /* **************  Features size and memory  ************** */

  // Host feature matrices size and memory 
  unsigned int size_Img_X[NB_LAYERS]; // Image X axis during all layers 
  unsigned int size_Img_Y[NB_LAYERS]; // Image Y axis during all layers 
  unsigned int size_feature[NB_LAYERS]; 
  unsigned int size_pooled[NB_LAYERS]; // Downsampled
  unsigned int size_unpooled[NB_LAYERS]; // Upsampled
    
  unsigned int mem_feature_encoder_conv[NB_CONVs]; 
  unsigned int mem_feature_decoder_conv[NB_CONVs]; 
    
  unsigned int mem_size_pooled[NB_LAYERS]; // Downsampled
  unsigned int mem_size_unpooled[NB_LAYERS]; // Upsampled
  
  unsigned pooled_index[NB_LAYERS] = {1, 3, 6, 9, 12};
  
  size_Img_X[0] = dimsImg.x; // Image X axis
  size_Img_Y[0]  = dimsImg.y; // Image Y axis
  size_feature[0] = dimsImg.x * dimsImg.y; // Feature
  size_pooled[0] =  size_feature[0] / (DOWN_SAMPLING*DOWN_SAMPLING);
  size_unpooled[0] =  size_feature[0];
  
   for (int j = 1; j < NB_LAYERS; j++){
  	size_Img_X[j] = size_Img_X[j-1] / DOWN_SAMPLING; // Image X axis
  	size_Img_Y[j]  = size_Img_Y[j-1] / DOWN_SAMPLING; // Image Y axis
    	size_feature[j] = size_Img_X[j] * size_Img_Y[j]; // Feature  
    	size_pooled[j] = size_feature[j] / (DOWN_SAMPLING*DOWN_SAMPLING); 
    	size_unpooled[j] = size_feature[j]; 
   }
  
   for (int j = 0; j < NB_CONVs; j++){
     	mem_feature_encoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]];
     	mem_feature_decoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]]; 
  }
  
  
   for (int j = 0; j < NB_LAYERS; j++){
   	mem_size_pooled[j] = sizeof(float) * size_pooled[j]; // Downsampled
   	mem_size_unpooled[j] = sizeof(float) * size_unpooled[j]; // Upsampled
   }


  /* **************  Input image  ************** */
    
  // Characteristics of the image to load
  int width_img, height_img, channels_img;
  
  // Load image
  unsigned char *h_input_img = stbi_load(INPUT_IMAGE_ADDR, &width_img, &height_img, &channels_img, 0); 
	
  if(h_input_img == NULL){
  	fprintf(stderr, "Failed to load the image!\n");
	exit(EXIT_FAILURE);
  }
  	
  // // printf("Loaded image with width %dpx height %dpx and channels %dpx \n", width_img, height_img, channels_img);
  	  
  // Define size and memory space for matrices (before resizing takes place)
  unsigned int size_input_Img = width_img * height_img * channels_img; // Image	
  unsigned int mem_size_input_Img = sizeof(uint8_t) * size_input_Img;	
  
  float x_conv = width_img/dimsImg.x;
  float y_conv = height_img/dimsImg.y;

  uint8_t *h_output_classes, *h_colored;
  checkCudaErrors(cudaMallocHost(&h_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMallocHost(&h_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));
  
  if (h_output_classes == NULL){
    fprintf(stderr, "Failed to allocate pixels class host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  if (h_colored == NULL){
    fprintf(stderr, "Failed to allocate colored host matrix!\n");
    exit(EXIT_FAILURE);
  }

  // Stream creation	
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));
  		
  		
  /* **************  Device Variables Section  ************** */
  
  // Allocate device memory for image convolution
  unsigned int size_resized_input_img = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS;	
  unsigned int mem_size_resized_input_img = sizeof(uint8_t) * size_resized_input_img;
  
  uint8_t *d_input_img, *d_input_img_resized; 
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img), mem_size_input_Img));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img_resized), mem_size_resized_input_img));


  // Filter weights and bias for the device  
  float *d_wino_filters_encoder[NB_CONVs];
  float *d_wino_filters_decoder[NB_CONVs];

  for (int j = 0; j < NB_CONVs; j++){
  	checkCudaErrors(cudaMalloc((&d_wino_filters_encoder[j]), mem_size_wino_filters_encoder[j]));
  	checkCudaErrors(cudaMalloc((&d_wino_filters_decoder[j]), mem_size_wino_filters_decoder[j]));
  }

  // Normalization mean, variance, weights and bias for the device  
  float *d_mean_encoder[NB_CONVs];
  float *d_var_encoder[NB_CONVs];
  float *d_weights_encoder[NB_CONVs];
  float *d_bias_norm_encoder[NB_CONVs];
  
  float *d_mean_decoder[NB_CONVs];
  float *d_var_decoder[NB_CONVs];
  float *d_weights_decoder[NB_CONVs];
  float *d_bias_norm_decoder[NB_CONVs];
  
  
  for (int j = 0; j < NB_CONVs; j++){
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_mean_encoder[j]), mem_size_norm_encoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_var_encoder[j]), mem_size_norm_encoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_weights_encoder[j]), mem_size_norm_encoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_bias_norm_encoder[j]), mem_size_norm_encoder[j]));
  	
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_mean_decoder[j]), mem_size_norm_decoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_var_decoder[j]), mem_size_norm_decoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_weights_decoder[j]), mem_size_norm_decoder[j]));
  	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_bias_norm_decoder[j]), mem_size_norm_decoder[j]));
  } 
			
  float *d_features_encoder_l[NB_CONVs];

    for (int j = 0; j < NB_CONVs; j++)
	checkCudaErrors(cudaMalloc((&d_features_encoder_l[j]), fps_encoder[j].out_channels*mem_feature_encoder_conv[j]));
  
  // Allocate device memory for downsampling
  float *d_max_pooled[NB_LAYERS];
  unsigned *d_max_pooled_idx[NB_LAYERS];

   for (int j = 0; j < NB_LAYERS; j++){
	checkCudaErrors(cudaMalloc((&d_max_pooled[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));
	checkCudaErrors(cudaMalloc((&d_max_pooled_idx[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));	
   }
		
  float *d_features_decoder_l[NB_CONVs];
  
  for (int j = 0; j < NB_CONVs; j++)
  	checkCudaErrors(cudaMalloc((&d_features_decoder_l[j]), fps_decoder[j].out_channels*mem_feature_decoder_conv[j]));
   
  // Allocate device memory for upsampling
  float *d_max_unpooled[NB_LAYERS];
   for (int j = 0; j < NB_LAYERS; j++)
	checkCudaErrors(cudaMalloc((&d_max_unpooled[j]), fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]));
	

  // Segmentation memory allocation
  uint8_t *d_output_classes, *d_colored;
  unsigned *d_class_count;
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output_classes), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_colored), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_class_count), sizeof(unsigned)*NB_CLASSES));  



  /* **************  Setup execution parameters  ************** */
       
  unsigned threads_per_block_resize = 512; 
  unsigned blocks_per_grid_resize = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS/threads_per_block_resize;
	  
  unsigned threads_per_block_conv[NB_LAYERS] = {64, 128, 256, 512, 512};
  unsigned blocks_per_grid_conv[NB_CONVs];
  
  for (int i = 0; i < NB_CONVs; i++){
  	// Check if Y axis is even w.r.t the Winograd tile size
  	if(size_Img_Y[CONV_2_LAYER[i]] % 2 == 0)
  		blocks_per_grid_conv[i] = ceil((fps_encoder[i].out_channels*size_Img_X[CONV_2_LAYER[i]]*size_Img_Y[CONV_2_LAYER[i]])/(WINO_TILE_SIZE*threads_per_block_conv[CONV_2_LAYER[i]])); 
	else
		blocks_per_grid_conv[i] = ceil((fps_encoder[i].out_channels*size_Img_X[CONV_2_LAYER[i]]*(size_Img_Y[CONV_2_LAYER[i]]+1))/(WINO_TILE_SIZE*threads_per_block_conv[CONV_2_LAYER[i]]));
  
  }

  unsigned threads_per_block_pool = 512; 
  unsigned blocks_per_grid_pool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_pool[i] = ceil(((size_Img_X[i]/DOWN_SAMPLING)*(size_Img_Y[i]/DOWN_SAMPLING)*fps_encoder[pooled_index[i]].out_channels)/threads_per_block_pool);
 	
 	
  unsigned threads_per_block_unpool = 512; 
  unsigned blocks_per_grid_unpool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_unpool[i] = ceil(size_Img_X[i]*size_Img_Y[i]*fps_encoder[pooled_index[i]].out_channels) / threads_per_block_unpool;
  

  int gridCols = ceil(float(size_Img_X[0]) / float(BLOCK_SIZE)); // size_Img_X = segmentation X length
  int gridRows = ceil(float(size_Img_Y[0]) / float(BLOCK_SIZE));

  unsigned threads_per_block_arg_max = MAX_THREADS_SIZE; 
  unsigned blocks_per_grid_arg_max = ceil((size_Img_X[0]*size_Img_Y[0])/threads_per_block_arg_max);
     
  
  /* **************  H2D Transfer  ************** */
	  
  // Input image (not resized)
  checkCudaErrors(cudaMemcpyAsync(d_input_img, h_input_img, mem_size_input_Img, cudaMemcpyHostToDevice, m0_stream));

  // Filters, bias, normalization
  
  for (int j = 0; j < NB_CONVs; j++){
 	checkCudaErrors(cudaMemcpyAsync(d_wino_filters_encoder[j], h_wino_filters_encoder[j], mem_size_wino_filters_encoder[j], cudaMemcpyHostToDevice, m0_stream)); 
 	checkCudaErrors(cudaMemcpyAsync(d_wino_filters_decoder[j], h_wino_filters_decoder[j], mem_size_wino_filters_decoder[j], cudaMemcpyHostToDevice, m0_stream)); 

  	checkCudaErrors(cudaMemcpyAsync(d_mean_encoder[j], h_mean_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_var_encoder[j], h_var_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_weights_encoder[j], h_weights_norm_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_bias_norm_encoder[j], h_bias_norm_encoder[j], mem_size_norm_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	
  	checkCudaErrors(cudaMemcpyAsync(d_mean_decoder[j], h_mean_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_var_decoder[j], h_var_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_weights_decoder[j], h_weights_norm_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_bias_norm_decoder[j], h_bias_norm_decoder[j], mem_size_norm_decoder[j], cudaMemcpyHostToDevice, m0_stream));

  }
  
  
  /* **************  Segnet execution  ************** */
  
  // Allocate CUDA timing events
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  unsigned layer_num = 0;
  unsigned conv_num = 0;

  // The times Segnet is run
  unsigned nIter = 1000; 
  
  // Execute Segnet compute kernels nIter times
  for (int j = 0; j < nIter; j++){
  
    	#ifdef EVENT_RECORD_CYCLES 
  		// Get GPU time stamp value for the first time
		getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
  	#endif
  	
        // Record the start event
        checkCudaErrors(cudaEventRecord(start, m0_stream));
  
  	// Reset arrays
        for (int j = 0; j < NB_LAYERS; j++){
		cudaMemset(&d_max_pooled[j], 0, fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]);
		cudaMemset(&d_max_unpooled[j], 0, fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]);	
   	}
   	
	// Resize input image
	resizeImgCUDA <<<blocks_per_grid_resize, threads_per_block_resize, 0, m0_stream>>> (d_input_img_resized, size_Img_X[0], size_Img_Y[0], d_input_img, width_img, NB_CHANNELS, x_conv, y_conv);

	// Wait for the image to be resized 
	cudaDeviceSynchronize(); 
		
  /* **************  Encoder  ************** */	
  	/* **************  Layer 1 - Convolution 1 ************** */	
  	conv_num = 0;
  	layer_num = 0;
	
	// Perform image*filter convolution for the resized image
  	wino_conv2x2_norm_ReLu_CUDA<64, 3> <<<size_Img_X[layer_num]*size_Img_Y[layer_num]/WINO_TILE_SIZE, 64, 0, m0_stream>>>(d_features_encoder_l[conv_num], d_input_img_resized, d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
  		
  	/* **************  Layer 1 - Convolution 2 ************** */	
	conv_num = 1;

	// Perform image*filter convolution 
	wino_conv2x2_norm_ReLu_CUDA<64, 64> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);

	// Downsample and save max values indices
	maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);		
			

	/* **************  Layer 2/3/4/5 - Convolution 3-13 ************** */
	for (int cnt_layer = 1; cnt_layer < NB_LAYERS; cnt_layer++){

	  	/* **************  Convolution ************** */
		conv_num++;	
		layer_num++;	 	
	  		
		// Perform image*filter convolution for the current feature
		if(layer_num < 2)
			wino2_conv2x2_norm_ReLu_CUDA<64, 64> <<<2*blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if(layer_num < 3)
			wino2_conv2x2_norm_ReLu_CUDA<128, 128> <<<2*blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if (layer_num < 4)
			wino2_conv2x2_norm_ReLu_CUDA<256, 256> <<< 2*blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if (layer_num < 5)
			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
					

		/* **************  Convolution  ************** */
			
		// Perform image*filter convolution for the current feature
		// CONV_PER_LAYER[cnt_layer]-1 because we are already doing one considering the downsampled matrices
		for (int cnt_conv_layer = 0; cnt_conv_layer < CONV_PER_LAYER[cnt_layer]-1; cnt_conv_layer++){
			conv_num++;
	
		if(layer_num < 2)
			wino_conv2x2_norm_ReLu_CUDA<128, 128> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if(layer_num < 3)
			wino_conv2x2_norm_ReLu_CUDA<256, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if (layer_num < 4)
			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
		else if (layer_num < 5)
			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_mean_encoder[conv_num], d_var_encoder[conv_num], d_weights_encoder[conv_num], d_bias_norm_encoder[conv_num], EPSILON);
			
		}
			
	
		// Downsample and save max values indices
		maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);
				
	 }
	  		
	  	
	/* **************  Decoder  ************** */
	/* **************  Layer 5/4/3/2/1 - Convolution 13-1 ************** */
	for (int cnt_layer = NB_LAYERS-1; cnt_layer >= 0; cnt_layer--, layer_num--){
	  	/* **************  Upsampling  ************** */	 	
		if(cnt_layer == NB_LAYERS-1){		 		
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_max_pooled[layer_num], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);	
		}else{
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_features_decoder_l[conv_num+1], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);
		}
				
		/* **************  Deconvolution  ************** */	
		// Perform image*filter convolution for the current feature
		if(cnt_layer > 2){
			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 1){
			wino_conv2x2_norm_ReLu_CUDA<256, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 0){
			wino_conv2x2_norm_ReLu_CUDA<128, 128> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else{
			wino_conv2x2_norm_ReLu_CUDA<64, 64> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);

		}
		
		// Next convolution
		conv_num--;		

		/* **************  Deconvolution Loop ************** */
		if(cnt_layer > 3){
			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);

			conv_num--;

			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		
		}else if(cnt_layer > 2){
			wino_conv2x2_norm_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
			
			conv_num--;
			

			wino3_conv2x2_norm_ReLu_CUDA<256, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 1){
			wino_conv2x2_norm_ReLu_CUDA<256, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
			
			conv_num--;
			
			wino3_conv2x2_norm_ReLu_CUDA<128, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else if(cnt_layer > 0){
			wino3_conv2x2_norm_ReLu_CUDA<64, 128> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);
		}else{
		// Check when 512/fps_decoder[conv_num].out_channels is not multiple. I may need a condition in the kernel
			wino4_conv2x2_norm_ReLu_CUDA <<<ceil((float)(NB_CLASSES*size_Img_X[layer_num]*size_Img_Y[layer_num]/(float)(WINO_TILE_SIZE*512))), 512, 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_mean_decoder[conv_num], d_var_decoder[conv_num], d_weights_decoder[conv_num], d_bias_norm_decoder[conv_num], EPSILON);

		}
		
			// Next convolution
			conv_num--;			
	 }

	
	
	/* **************  Segmentation  ************** */		

	// Get the index (class) with the highest value for each pixel
	argMax3D_CUDA<<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_output_classes, d_features_decoder_l[0], NB_CLASSES, size_feature[0]);	
				
	/* **************  Coloring  ************** */
	
	// Array with the color code for each class
	uint8_t d_class_tag_val[NB_CHANNELS];
	
	for(int i = 0; i < NB_CLASSES; i++){
	
		// Background clutter Class 0
		if(i == 0){ 
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;		
		// Building Class 1
		}else if(i == 1){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;
		// Road Class 2
		}else if(i == 2){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 128;
		// Static_Car Class 3
		}else if(i == 3){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 128;
		// Tree Class 4
		}else if (i== 4){
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Vegetation Class 5
		}else if (i == 5){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Human Class 6
		}else if (i == 6){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 0;
		}

		// Color each pixel according to its class
		createRGBclassesCUDA <<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_colored, d_class_count, d_output_classes, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, i, d_class_tag_val[0], d_class_tag_val[1], d_class_tag_val[2]);
		
	}
	
	
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, m0_stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal;                        
  printf( "Time: %.3f msec \n", msecPerMatrixMul);  
  
  #ifdef EVENT_RECORD_CYCLES 	
	  // Wait for the end of execution of all the threads blocks
	  cudaDeviceSynchronize();   
	  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();   
			    
	  // Calculate execution time in cycles and display it	  
	  calculate_time_diff_clock(1); 
	  print_time_diff_clock();
  #endif
		
  }

  // Transfer of the matrix with the colored pixels 
  checkCudaErrors(cudaMemcpy(h_colored, d_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS, cudaMemcpyDeviceToHost)); 
	    	
  // Export colored segmentation image
  stbi_write_png(OUTPUT_IMAGE_ADDR, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, h_colored, size_Img_X[0]*NB_CHANNELS);

  // Transfer of the matrix with the pixel classification 
  checkCudaErrors(cudaMemcpy(h_output_classes, d_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0], cudaMemcpyDeviceToHost)); 

  // Wait until all computations and transactions have finished for stream m0_stream
  checkCudaErrors(cudaStreamSynchronize(m0_stream));  
  
  #ifdef EVENT_RECORD_CYCLES 
   	// Clean up clock memory 
  	clean_clock();
  #endif
  
  // Host
  stbi_image_free(h_input_img);

   for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFreeHost(h_filters_encoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_filters_decoder[cnt]));

   	checkCudaErrors(cudaFreeHost(h_mean_encoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_var_encoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_weights_norm_encoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_bias_norm_encoder[cnt]));
   	
   	checkCudaErrors(cudaFreeHost(h_mean_decoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_var_decoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_weights_norm_decoder[cnt]));
   	checkCudaErrors(cudaFreeHost(h_bias_norm_decoder[cnt]));
  }
  
  checkCudaErrors(cudaFreeHost(h_output_classes));   
  checkCudaErrors(cudaFreeHost(h_colored));
      
  // Device
  checkCudaErrors(cudaFree(d_input_img));
  checkCudaErrors(cudaFree(d_input_img_resized));
  

  for(int cnt = 0; cnt < NB_CONVs; cnt++){
   	checkCudaErrors(cudaFree(d_mean_encoder[cnt]));
   	checkCudaErrors(cudaFree(d_var_encoder[cnt]));
   	checkCudaErrors(cudaFree(d_weights_encoder[cnt]));
   	checkCudaErrors(cudaFree(d_bias_norm_encoder[cnt]));
   	
   	checkCudaErrors(cudaFree(d_mean_decoder[cnt]));
   	checkCudaErrors(cudaFree(d_var_decoder[cnt]));
   	checkCudaErrors(cudaFree(d_weights_decoder[cnt]));
   	checkCudaErrors(cudaFree(d_bias_norm_decoder[cnt]));
   	
  	checkCudaErrors(cudaFree(d_features_encoder_l[cnt]));
  	checkCudaErrors(cudaFree(d_features_decoder_l[cnt]));	 	
  }
  
  
  // Upsampling/downsampling
  for(int cnt = 0; cnt < NB_LAYERS; cnt++){
  	
	checkCudaErrors(cudaFree(d_max_pooled[cnt]));
	checkCudaErrors(cudaFree(d_max_pooled_idx[cnt]));
	checkCudaErrors(cudaFree(d_max_unpooled[cnt]));
  }
  
  

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

   
}



/* SegnetImplementationIntegratedWinograd2x2Arch1
 *
 * Description: Executes the Segnet CNN considering multiple enconder-decoder layers.
 *		 Convolutions are made via Winograd. The filter weights integrate the 
 *		 batch normalization. Float32 is used.	
 *
 * Parameter:   
 *		- const dim3 &dimsImg: Dimensions of the resized input image 
 *		- const dim3 &dimsFilter: Dimensions of the filters
 *		- const char INPUT_IMAGE_ADDR[]: Path to input image
 *		- const char BIN_BASE_ADDR[]: Path to binary files directory
 *		- const char OUTPUT_IMAGE_ADDR[]: Path to output image
 *
 * Returns:     Nothing
 *
 * */
void SegnetImplementationIntegratedWinograd2x2Arch1(const dim3 &dimsImg, const dim3 &dimsFilter, const char INPUT_IMAGE_ADDR[], const char BIN_BASE_ADDR[], const char OUTPUT_IMAGE_ADDR[]){
 
  #ifdef EVENT_RECORD_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
  #endif
                 
  // Number of encoder-decoder layers 
  int const NB_LAYERS = 5;
  // Horizontal/Vertical downsampling/upsampling value (total size change is given by DOWN_SAMPLING*DOWN_SAMPLING)
  int const DOWN_SAMPLING = 2;
  // Number of classes
  int const NB_CLASSES = 7;
  // Number of channels
  int const NB_CHANNELS = 3;
  // Number of convolutions in this Segent implementation
  const unsigned NB_CONVs = 13;
  // Output Winograd convolution tile dimension
  const unsigned WINO_TILE_SIZE = 4;

  // Convolutions per encoder/decoder layer
  const unsigned NB_CONV_LAYER1 = 2;
  const unsigned NB_CONV_LAYER2 = 2;
  const unsigned NB_CONV_LAYER3 = 3;
  const unsigned NB_CONV_LAYER4 = 3;
  const unsigned NB_CONV_LAYER5 = 3;

  const unsigned CONV_2_LAYER[NB_CONVs] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  const unsigned CONV_PER_LAYER[NB_LAYERS] = {NB_CONV_LAYER1, NB_CONV_LAYER2, NB_CONV_LAYER3, NB_CONV_LAYER4, NB_CONV_LAYER5};
  
  
  /* **************  Weights, bias and normalization data extraction  ************** */
  
  char bin_file_path_en_conv[NB_CONVs][128];
  char bin_file_path_de_conv[NB_CONVs][128];
  
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	strcpy(bin_file_path_en_conv[cnt], BIN_BASE_ADDR);
  	strcpy(bin_file_path_de_conv[cnt], BIN_BASE_ADDR);
  }
  
  strcat(bin_file_path_en_conv[0], "11_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[1], "12_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[2], "21_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[3], "22_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[4], "31_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[5], "32_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[6], "33_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[7], "41_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[8], "42_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[9], "43_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[10], "51_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[11], "52_EncoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_en_conv[12], "53_EncoderIntegratedConvAndBatchNorm.bin");  

  strcat(bin_file_path_de_conv[0], "11_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[1], "12_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[2], "21_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[3], "22_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[4], "31_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[5], "32_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[6], "33_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[7], "41_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[8], "42_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[9], "43_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[10], "51_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[11], "52_DecoderIntegratedConvAndBatchNorm.bin");
  strcat(bin_file_path_de_conv[12], "53_DecoderIntegratedConvAndBatchNorm.bin"); 
   
   
  struct filter_prop_struct fps_encoder[NB_CONVs], fps_decoder[NB_CONVs];
  
  float *h_filters_encoder[NB_CONVs];
  float *h_bias_encoder[NB_CONVs];
  
  float *h_filters_decoder[NB_CONVs];
  float *h_bias_decoder[NB_CONVs];
  
  // Extract filters weights and bias
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	load_filters_integrated(&h_filters_encoder[cnt], &h_bias_encoder[cnt], &fps_encoder[cnt], bin_file_path_en_conv[cnt]);
  	load_filters_integrated(&h_filters_decoder[cnt], &h_bias_decoder[cnt], &fps_decoder[cnt], bin_file_path_de_conv[cnt]);
  }
  

  float *h_wino_filters_encoder[NB_CONVs];
  float *h_wino_filters_decoder[NB_CONVs];
  
  // Obtain Winograd transformations for filters
  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	wino_3x3filter_2x2transform(&h_wino_filters_encoder[cnt], h_filters_encoder[cnt], fps_encoder[cnt]);
  	wino_3x3filter_2x2transform(&h_wino_filters_decoder[cnt], h_filters_decoder[cnt], fps_decoder[cnt]);
  }

       		
  // Encoder 
  unsigned size_wino_filters_encoder[NB_CONVs];
  unsigned size_bias_encoder[NB_CONVs];
  	
  unsigned mem_size_wino_filters_encoder[NB_CONVs];
  unsigned mem_size_bias_encoder[NB_CONVs];

  // Decoder
  unsigned size_wino_filters_decoder[NB_CONVs];
  unsigned size_bias_decoder[NB_CONVs];
  	
  unsigned mem_size_wino_filters_decoder[NB_CONVs];
  unsigned mem_size_bias_decoder[NB_CONVs];
   
    for (int j = 0; j < NB_CONVs; j++){
    	// Encoder
    	size_wino_filters_encoder[j] = fps_encoder[j].in_channels * fps_encoder[j].out_channels * 16; // for 3x3 filters and 2x2 output convolution tile
    	size_bias_encoder[j] = fps_encoder[j].out_channels;
    	
    	mem_size_wino_filters_encoder[j] = sizeof(float) * size_wino_filters_encoder[j];
    	mem_size_bias_encoder[j] = sizeof(float) * size_bias_encoder[j];
    		
    	// Decoder
    	size_wino_filters_decoder[j] = fps_decoder[j].in_channels * fps_decoder[j].out_channels * 16; // for 3x3 filters and 2x2 output convolution tile
    	size_bias_decoder[j] = fps_decoder[j].out_channels;
    	
    	mem_size_wino_filters_decoder[j] = sizeof(float) * size_wino_filters_decoder[j];
    	mem_size_bias_decoder[j] = sizeof(float) * size_bias_decoder[j];
    }
    

  
  /* **************  Features size and memory  ************** */

  // Host feature matrices size and memory 
  unsigned int size_Img_X[NB_LAYERS]; // Image X axis during all layers 
  unsigned int size_Img_Y[NB_LAYERS]; // Image Y axis during all layers 
  unsigned int size_feature[NB_LAYERS]; 
  unsigned int size_pooled[NB_LAYERS]; // Downsampled
  unsigned int size_unpooled[NB_LAYERS]; // Upsampled
    
  unsigned int mem_feature_encoder_conv[NB_CONVs]; 
  unsigned int mem_feature_decoder_conv[NB_CONVs]; 
    
  unsigned int mem_size_pooled[NB_LAYERS]; // Downsampled
  unsigned int mem_size_unpooled[NB_LAYERS]; // Upsampled
  
  unsigned pooled_index[NB_LAYERS] = {1, 3, 6, 9, 12};
  
  size_Img_X[0] = dimsImg.x; // Image X axis
  size_Img_Y[0]  = dimsImg.y; // Image Y axis
  size_feature[0] = dimsImg.x * dimsImg.y; // Feature
  size_pooled[0] =  size_feature[0] / (DOWN_SAMPLING*DOWN_SAMPLING);
  size_unpooled[0] =  size_feature[0];
  
   for (int j = 1; j < NB_LAYERS; j++){
  	size_Img_X[j] = size_Img_X[j-1] / DOWN_SAMPLING; // Image X axis
  	size_Img_Y[j]  = size_Img_Y[j-1] / DOWN_SAMPLING; // Image Y axis
    	size_feature[j] = size_Img_X[j] * size_Img_Y[j]; // Feature  
    	size_pooled[j] = size_feature[j] / (DOWN_SAMPLING*DOWN_SAMPLING); 
    	size_unpooled[j] = size_feature[j]; 
   }
  
   for (int j = 0; j < NB_CONVs; j++){
     	mem_feature_encoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]];
     	mem_feature_decoder_conv[j] = sizeof(float) * size_feature[CONV_2_LAYER[j]]; 
  }
  
  
   for (int j = 0; j < NB_LAYERS; j++){
   	mem_size_pooled[j] = sizeof(float) * size_pooled[j]; // Downsampled
   	mem_size_unpooled[j] = sizeof(float) * size_unpooled[j]; // Upsampled
   }


  /* **************  Input image  ************** */
    
  // Characteristics of the image to load
  int width_img, height_img, channels_img;
  
  // Load image
  unsigned char *h_input_img = stbi_load(INPUT_IMAGE_ADDR, &width_img, &height_img, &channels_img, 0); 
	
  if(h_input_img == NULL){
  	fprintf(stderr, "Failed to load the image!\n");
	exit(EXIT_FAILURE);
  }
  	
  // printf("Loaded image with width %dpx height %dpx and channels %dpx \n", width_img, height_img, channels_img);
  	  
  // Define size and memory space for matrices (before resizing takes place)
  unsigned int size_input_Img = width_img * height_img * channels_img; // Image	
  unsigned int mem_size_input_Img = sizeof(uint8_t) * size_input_Img;	
  
  float x_conv = width_img/dimsImg.x;
  float y_conv = height_img/dimsImg.y;

  uint8_t *h_output_classes, *h_colored;
  checkCudaErrors(cudaMallocHost(&h_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMallocHost(&h_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));
  
  if (h_output_classes == NULL){
    fprintf(stderr, "Failed to allocate pixels class host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  if (h_colored == NULL){
    fprintf(stderr, "Failed to allocate colored host matrix!\n");
    exit(EXIT_FAILURE);
  }

  // Stream creation	
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));
  		
  		
  /* **************  Device Variables Section  ************** */
  
  // Allocate device memory for image convolution
  unsigned int size_resized_input_img = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS;	
  unsigned int mem_size_resized_input_img = sizeof(uint8_t) * size_resized_input_img;
  
  uint8_t *d_input_img, *d_input_img_resized; 
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img), mem_size_input_Img));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input_img_resized), mem_size_resized_input_img));


  // Filter weights and bias for the device  
  float *d_bias_encoder[NB_CONVs];
  float *d_bias_decoder[NB_CONVs]; 
  float *d_wino_filters_encoder[NB_CONVs];
  float *d_wino_filters_decoder[NB_CONVs];

  for (int j = 0; j < NB_CONVs; j++){
  	checkCudaErrors(cudaMalloc((&d_bias_encoder[j]), mem_size_bias_encoder[j]));
  	checkCudaErrors(cudaMalloc((&d_bias_decoder[j]), mem_size_bias_decoder[j]));
  		
  	checkCudaErrors(cudaMalloc((&d_wino_filters_encoder[j]), mem_size_wino_filters_encoder[j]));
  	checkCudaErrors(cudaMalloc((&d_wino_filters_decoder[j]), mem_size_wino_filters_decoder[j]));
  }
			
  float *d_features_encoder_l[NB_CONVs];

    for (int j = 0; j < NB_CONVs; j++)
	checkCudaErrors(cudaMalloc((&d_features_encoder_l[j]), fps_encoder[j].out_channels*mem_feature_encoder_conv[j]));
  
  // Allocate device memory for downsampling
  float *d_max_pooled[NB_LAYERS];
  unsigned *d_max_pooled_idx[NB_LAYERS];

   for (int j = 0; j < NB_LAYERS; j++){
	checkCudaErrors(cudaMalloc((&d_max_pooled[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));
	checkCudaErrors(cudaMalloc((&d_max_pooled_idx[j]), fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]));	
   }
		
  float *d_features_decoder_l[NB_CONVs];
  
  for (int j = 0; j < NB_CONVs; j++)
  	checkCudaErrors(cudaMalloc((&d_features_decoder_l[j]), fps_decoder[j].out_channels*mem_feature_decoder_conv[j]));
   
  // Allocate device memory for upsampling
  float *d_max_unpooled[NB_LAYERS];
   for (int j = 0; j < NB_LAYERS; j++)
	checkCudaErrors(cudaMalloc((&d_max_unpooled[j]), fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]));
	

  // Segmentation memory allocation
  uint8_t *d_output_classes, *d_colored;
  unsigned *d_class_count;
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output_classes), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_colored), sizeof(uint8_t)* size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS));  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_class_count), sizeof(unsigned)*NB_CLASSES));  


  /* **************  Setup execution parameters  ************** */
       
  unsigned threads_per_block_resize = 512; 
  unsigned blocks_per_grid_resize = size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS/threads_per_block_resize;
	  
  unsigned threads_per_block_conv[NB_LAYERS] = {64, 128, 256, 512, 512};
  unsigned blocks_per_grid_conv[NB_CONVs];
  
  for (int i = 0; i < NB_CONVs; i++){
  	// Check if Y axis is even w.r.t the Winograd tile size
  	if(size_Img_Y[CONV_2_LAYER[i]] % 2 == 0)
  		blocks_per_grid_conv[i] = ceil((fps_encoder[i].out_channels*size_Img_X[CONV_2_LAYER[i]]*size_Img_Y[CONV_2_LAYER[i]])/(WINO_TILE_SIZE*threads_per_block_conv[CONV_2_LAYER[i]])); 
	else
		blocks_per_grid_conv[i] = ceil((fps_encoder[i].out_channels*size_Img_X[CONV_2_LAYER[i]]*(size_Img_Y[CONV_2_LAYER[i]]+1))/(WINO_TILE_SIZE*threads_per_block_conv[CONV_2_LAYER[i]]));
  
  }

  unsigned threads_per_block_pool = 512; 
  unsigned blocks_per_grid_pool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_pool[i] = ceil(((size_Img_X[i]/DOWN_SAMPLING)*(size_Img_Y[i]/DOWN_SAMPLING)*fps_encoder[pooled_index[i]].out_channels)/threads_per_block_pool);
 	
 	
  unsigned threads_per_block_unpool = 512; 
  unsigned blocks_per_grid_unpool[NB_LAYERS];

  for (int i = 0; i < NB_LAYERS; i++)
 	blocks_per_grid_unpool[i] = ceil(size_Img_X[i]*size_Img_Y[i]*fps_encoder[pooled_index[i]].out_channels) / threads_per_block_unpool;
  

  int gridCols = ceil(float(size_Img_X[0]) / float(BLOCK_SIZE)); // size_Img_X = segmentation X length
  int gridRows = ceil(float(size_Img_Y[0]) / float(BLOCK_SIZE));

  unsigned threads_per_block_arg_max = MAX_THREADS_SIZE; 
  unsigned blocks_per_grid_arg_max = ceil((size_Img_X[0]*size_Img_Y[0])/threads_per_block_arg_max);
     
  
  /* **************  H2D Transfer  ************** */
	  
  // Input image (not resized)
  checkCudaErrors(cudaMemcpyAsync(d_input_img, h_input_img, mem_size_input_Img, cudaMemcpyHostToDevice, m0_stream));

  // Filters, bias, normalization
  for (int j = 0; j < NB_CONVs; j++){
  	checkCudaErrors(cudaMemcpyAsync(d_bias_encoder[j], h_bias_encoder[j], mem_size_bias_encoder[j], cudaMemcpyHostToDevice, m0_stream));
  	checkCudaErrors(cudaMemcpyAsync(d_bias_decoder[j], h_bias_decoder[j], mem_size_bias_decoder[j], cudaMemcpyHostToDevice, m0_stream));
  	
 	checkCudaErrors(cudaMemcpyAsync(d_wino_filters_encoder[j], h_wino_filters_encoder[j], mem_size_wino_filters_encoder[j], cudaMemcpyHostToDevice, m0_stream)); 
 	checkCudaErrors(cudaMemcpyAsync(d_wino_filters_decoder[j], h_wino_filters_decoder[j], mem_size_wino_filters_decoder[j], cudaMemcpyHostToDevice, m0_stream)); 

  }
  
  
  /* **************  Segnet execution  ************** */
  
  // Allocate CUDA timing events
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  unsigned layer_num = 0;
  unsigned conv_num = 0;

  // The times Segnet is run
  unsigned nIter = 1000; 

  // Execute Segnet compute kernels nIter times
  for (int j = 0; j < nIter; j++){
  
    	#ifdef EVENT_RECORD_CYCLES 
  		// Get GPU time stamp value for the first time
		getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
  	#endif
  	
  	// Record the start event
  	checkCudaErrors(cudaEventRecord(start, m0_stream));
  
  	// Reset arrays
        for (int j = 0; j < NB_LAYERS; j++){
		cudaMemset(&d_max_pooled[j], 0, fps_encoder[pooled_index[j]].out_channels*mem_size_pooled[j]);
		cudaMemset(&d_max_unpooled[j], 0, fps_decoder[pooled_index[j]].out_channels*mem_size_unpooled[j]);	
   	}
   	
	// Resize input image
	resizeImgCUDA <<<blocks_per_grid_resize, threads_per_block_resize, 0, m0_stream>>> (d_input_img_resized, size_Img_X[0], size_Img_Y[0], d_input_img, width_img, NB_CHANNELS, x_conv, y_conv);

	// Wait for the image to be resized 
	cudaDeviceSynchronize(); 
		
  	/* **************  Encoder  ************** */	
  	/* **************  Layer 1 - Convolution 1 ************** */	
  	conv_num = 0;
  	layer_num = 0;
	
	// Perform image*filter convolution for the resized image
  	wino_conv2x2_ReLu_CUDA<64, 3> <<<size_Img_X[layer_num]*size_Img_Y[layer_num]/WINO_TILE_SIZE, 64, 0, m0_stream>>>(d_features_encoder_l[conv_num], d_input_img_resized, d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
  		
  	/* **************  Layer 1 - Convolution 2 ************** */	
	conv_num = 1;

	// Perform image*filter convolution 
	wino_conv2x2_ReLu_CUDA<64, 64> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);

	// Downsample and save max values indices
	maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);		
			

	/* **************  Layer 2/3/4/5 - Convolution 3-13 ************** */
	for (int cnt_layer = 1; cnt_layer < NB_LAYERS; cnt_layer++){

	  	/* **************  Convolution ************** */
		conv_num++;	
		layer_num++;	 	
	  		
		// Perform image*filter convolution for the current feature
		if(layer_num < 2)
			wino2_conv2x2_ReLu_CUDA<64, 64> <<<2*blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
		else if(layer_num < 3)
			wino2_conv2x2_ReLu_CUDA<128, 128> <<<2*blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
		else if (layer_num < 4)
			wino2_conv2x2_ReLu_CUDA<256, 256> <<<2*blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);			
		else if (layer_num < 5)
			wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_max_pooled[layer_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
					

		/* **************  Convolution  ************** */
			
		// Perform image*filter convolution for the current feature
		// CONV_PER_LAYER[cnt_layer]-1 because we are already doing one considering the downsampled matrices
		for (int cnt_conv_layer = 0; cnt_conv_layer < CONV_PER_LAYER[cnt_layer]-1; cnt_conv_layer++){
			conv_num++;
	
			if(layer_num < 2)
				wino_conv2x2_ReLu_CUDA<128, 128> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
			else if(layer_num < 3)
				wino_conv2x2_ReLu_CUDA<256, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
			else if (layer_num < 4)
				wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
			else if (layer_num < 5)
				wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_encoder_l[conv_num], d_features_encoder_l[conv_num-1], d_wino_filters_encoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].in_channels, fps_encoder[conv_num].out_channels, d_bias_encoder[conv_num]);
			
		}
		
		
		// Downsample and save max values indices
		maxPooling_CUDA <DOWN_SAMPLING><<<blocks_per_grid_pool[layer_num], threads_per_block_pool, 0, m0_stream>>>(d_max_pooled[layer_num], d_max_pooled_idx[layer_num], d_features_encoder_l[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_encoder[conv_num].out_channels);
		
				
	 }
	  		
  	
	/* **************  Decoder  ************** */
	/* **************  Layer 5/4/3/2/1 - Convolution 13-1 ************** */
	
	for (int cnt_layer = NB_LAYERS-1; cnt_layer >= 0; cnt_layer--, layer_num--){
	  	/* **************  Upsampling  ************** */	 	
		if(cnt_layer == NB_LAYERS-1){		 		
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_max_pooled[layer_num], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);	
		}else{
		  	indicesUnpooling_CUDA <<<blocks_per_grid_unpool[layer_num], threads_per_block_unpool, 0, m0_stream>>>(d_max_unpooled[layer_num], d_features_decoder_l[conv_num+1], size_Img_X[layer_num]/DOWN_SAMPLING, size_Img_Y[layer_num]/DOWN_SAMPLING, fps_decoder[conv_num].in_channels, d_max_pooled_idx[layer_num]);
		}
				
		/* **************  Deconvolution  ************** */	
		// Perform image*filter convolution for the current feature
		if(cnt_layer > 2){
			wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
			
		}else if(cnt_layer > 1){
		wino_conv2x2_ReLu_CUDA<256, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		
		}else if(cnt_layer > 0){
		wino_conv2x2_ReLu_CUDA<128, 128> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		
		}else{
			wino_conv2x2_ReLu_CUDA<64, 64> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_max_unpooled[layer_num], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);

		}
		
		// Next convolution
		conv_num--;		

		/* **************  Deconvolution Loop ************** */
		if(cnt_layer > 3){
			wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);

			conv_num--;
			
			wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		
		}else if(cnt_layer > 2){
			wino_conv2x2_ReLu_CUDA<512, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
			
			conv_num--;
		
			wino3_conv2x2_ReLu_CUDA<256, 512> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else if(cnt_layer > 1){
			wino_conv2x2_ReLu_CUDA<256, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
			
			conv_num--;

			wino3_conv2x2_ReLu_CUDA<128, 256> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else if(cnt_layer > 0){
			wino3_conv2x2_ReLu_CUDA<64, 128> <<<blocks_per_grid_conv[conv_num], threads_per_block_conv[layer_num-1], 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}else{
			wino4_conv2x2_ReLu_CUDA <<<ceil((float)(NB_CLASSES*size_Img_X[layer_num]*size_Img_Y[layer_num]/(float)(WINO_TILE_SIZE*512))), 512, 0, m0_stream>>>(d_features_decoder_l[conv_num], d_features_decoder_l[conv_num+1], d_wino_filters_decoder[conv_num], size_Img_X[layer_num], size_Img_Y[layer_num], fps_decoder[conv_num].in_channels, fps_decoder[conv_num].out_channels, d_bias_decoder[conv_num]);
		}
		
		// Next convolution
		conv_num--;
				
	 }

	
	
	/* **************  Segmentation  ************** */		

	// Get the index (class) with the highest value for each pixel
	argMax3D_CUDA<<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_output_classes, d_features_decoder_l[0], NB_CLASSES, size_feature[0]);	
				
	/* **************  Coloring  ************** */
	
	// Array with the color code for each class
	uint8_t d_class_tag_val[NB_CHANNELS];
	
	for(int i = 0; i < NB_CLASSES; i++){
	
		// Background clutter Class 0
		if(i == 0){ 
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;		
		// Building Class 1
		}else if(i == 1){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 0;
		// Road Class 2
		}else if(i == 2){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 128;
		// Static_Car Class 3
		}else if(i == 3){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 0;
			d_class_tag_val[2] = 128;
		// Tree Class 4
		}else if (i== 4){
			d_class_tag_val[0] = 0;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Vegetation Class 5
		}else if (i == 5){ 
			d_class_tag_val[0] = 128;
			d_class_tag_val[1] = 128;
			d_class_tag_val[2] = 0;
		// Human Class 6
		}else if (i == 6){ 
			d_class_tag_val[0] = 64;
			d_class_tag_val[1] = 64;
			d_class_tag_val[2] = 0;
		}
		// Color each pixel according to its class
		createRGBclassesCUDA <<<blocks_per_grid_arg_max, threads_per_block_arg_max, 0, m0_stream>>>(d_colored, d_class_count, d_output_classes, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, i, d_class_tag_val[0], d_class_tag_val[1], d_class_tag_val[2]);
		
	}
	
	
  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, m0_stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal;                        
  printf( "Time: %.3f msec \n", msecPerMatrixMul);
		
  #ifdef EVENT_RECORD_CYCLES 	
	  // Wait for the end of execution of all the threads blocks
	  cudaDeviceSynchronize();   
	  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();   
			    
	  // Calculate execution time in cycles and display it	  
	  calculate_time_diff_clock(1); 
	  print_time_diff_clock();
  #endif
  }
 
  // Transfer of the matrix with the colored pixels 
  checkCudaErrors(cudaMemcpy(h_colored, d_colored, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0]*NB_CHANNELS, cudaMemcpyDeviceToHost)); 
	    	
  // Export colored segmentation image
  stbi_write_png(OUTPUT_IMAGE_ADDR, size_Img_X[0], size_Img_Y[0], NB_CHANNELS, h_colored, size_Img_X[0]*NB_CHANNELS);

  // Transfer of the matrix with the pixel classification 
  checkCudaErrors(cudaMemcpy(h_output_classes, d_output_classes, sizeof(uint8_t)*size_Img_X[0]*size_Img_Y[0], cudaMemcpyDeviceToHost)); 

  // Wait until all computations and transactions have finished for stream m0_stream
  checkCudaErrors(cudaStreamSynchronize(m0_stream));  
  
  #ifdef EVENT_RECORD_CYCLES 
   	// Clean up clock memory 
  	clean_clock();
  #endif
  
  // Host
  stbi_image_free(h_input_img);

   for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFreeHost(h_filters_encoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_filters_decoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_bias_encoder[cnt]));
  	checkCudaErrors(cudaFreeHost(h_bias_decoder[cnt]));
  }
  
  checkCudaErrors(cudaFreeHost(h_output_classes));   
  checkCudaErrors(cudaFreeHost(h_colored));
      
  // Device
  checkCudaErrors(cudaFree(d_input_img));
  checkCudaErrors(cudaFree(d_input_img_resized));
  

  for(int cnt = 0; cnt < NB_CONVs; cnt++){
  	checkCudaErrors(cudaFree(d_bias_encoder[cnt]));
  	checkCudaErrors(cudaFree(d_bias_decoder[cnt]));
   
  	checkCudaErrors(cudaFree(d_features_encoder_l[cnt]));
  	checkCudaErrors(cudaFree(d_features_decoder_l[cnt]));	 	
  }
  
  
  // Upsampling/downsampling
  for(int cnt = 0; cnt < NB_LAYERS; cnt++){
  	
	checkCudaErrors(cudaFree(d_max_pooled[cnt]));
	checkCudaErrors(cudaFree(d_max_pooled_idx[cnt]));
	checkCudaErrors(cudaFree(d_max_unpooled[cnt]));
  }
  
  

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

   
}



#endif

