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


/*--------------------------- segnet_utils.h -----------------------------
|  File segnet_utils.h
|
|  Description: Common functions for Segnet NN 
|
|  Version: 1.0
*-----------------------------------------------------------------------*/

// Compute kernel variables
unsigned const BLOCK_SIZE = 32;
unsigned const BLOCK_SIZE_256 = 256;
unsigned const SM_MIN_OPT_THREADS_SIZE = 128;
unsigned const MAX_THREADS_SIZE = 1024;
unsigned const BLOCK_2_SM_SIZE = 16;


struct filter_prop_struct{
  unsigned in_channels;
  unsigned out_channels;
  unsigned kernel_size_cols;
  unsigned kernel_size_rows;
};


/* load_filters
 *
 * Description: Loads the filter weights from binary files.
 *		 The data is divided in 12 unsigned variables of header
 *		 followed by a variable number of filter weight data 
 *		(input channels * output channels * filter width * filter height). 
 *
 * Parameter:   
 *		- float** out_filters: Pointer to filter weights (array of elements)
 *		- struct filter_prop_struct *fps: Structure where the filter specs are saved
 *		- const char* PATH: Path to input filter weights file 

 *
 * Returns:     Nothing
 *
 * */
void load_filters(float** out_filters, struct filter_prop_struct *fps, const char* PATH){

  std::ifstream binfile;
  binfile = std::ifstream(PATH, std::ios::in | std::ios::binary);

  unsigned length;
  float *buffer = NULL;
  
  float* filters;
  
  const unsigned DATA_OFFSET = 12;
 
  
    if(binfile){
	    // get length of file:
	    binfile.seekg (0, binfile.end);
	    length = binfile.tellg() / sizeof(float);
	    binfile.seekg (0, binfile.beg);

	    buffer = new float[length];
 
	    // read data 
	    binfile.read((char*)(buffer), sizeof(float)*length);

	    binfile.close();
    }else{
 	    fprintf(stderr, "Failed to locate the binary file!\n");
	    exit(EXIT_FAILURE);
    }
    
    fps->in_channels = buffer[0];      // Number of features
    fps->out_channels = buffer[1];     // Number of filters per feature
    fps->kernel_size_cols = buffer[2];    // Filter cols
    fps->kernel_size_rows = buffer[3];    // Filter rows
	
    unsigned total_filters_memory = sizeof(float)*(fps->in_channels * fps->out_channels * fps->kernel_size_cols * fps->kernel_size_rows);
    
    // Allocate memory
    checkCudaErrors(cudaMallocHost(&filters, total_filters_memory));


    if (filters == NULL){
	fprintf(stderr, "Failed to allocate filters host matrix!\n");
	exit(EXIT_FAILURE);
    }
 
    unsigned nb_features = fps->in_channels * fps->out_channels;	
	
    // Order data in form of rows, cols, depth, filters per feature
    for(int i = 0; i < fps->kernel_size_rows; i++)
    	for(int j = 0; j < fps->kernel_size_cols; j++)
    		for(int k = 0; k < fps->out_channels; k++)
			for(int l = 0; l < fps->in_channels; l++)
				filters[i*fps->kernel_size_cols*nb_features + j*nb_features + l*fps->out_channels + k] = buffer[i*fps->kernel_size_rows*fps->out_channels*fps->in_channels + j*fps->out_channels*fps->in_channels + k*fps->in_channels + l + DATA_OFFSET];
				
    delete[] buffer;

    *out_filters = filters;

}



/* load_filters_integrated
 *
 * Description: Loads the filter weights with the batch normalization inlcuded and 
 *		the convolution bias from binary files.
 *		 The data is divided in 12 unsigned variables of header
 *		 followed by a variable number of filter weight data 
 *		(input channels * output channels * filter width * filter height). 
 *
 * Parameter:   
 *		- float** out_filters: Pointer to filter weights (array of elements)
 *		- float** out_bias: Pointer to convolution bias (array of elements)
 *		- struct filter_prop_struct *fps: Structure where the filter specs are saved
 *		- const char* PATH: Path to input filter weights file 

 *
 * Returns:     Nothing
 *
 * */
void load_filters_integrated(float** out_filters, float** out_bias, struct filter_prop_struct *fps, const char* PATH){

    std::ifstream binfile;
  binfile = std::ifstream(PATH, std::ios::in | std::ios::binary);

  unsigned length;
  float *buffer = NULL;
  
  float* filters;
  float* bias;
  
  const unsigned DATA_OFFSET = 12;
 
  
    if(binfile){
	    // get length of file:
	    binfile.seekg (0, binfile.end);
	    length = binfile.tellg() / sizeof(float);
	    binfile.seekg (0, binfile.beg);

	    buffer = new float[length];

	    // read data 
	    binfile.read((char*)(buffer), sizeof(float)*length);

	    binfile.close();
    }else{
 	    fprintf(stderr, "Failed to locate the binary file!\n");
	    exit(EXIT_FAILURE);
    }

    fps->in_channels = buffer[0];      // Number of features
    fps->out_channels = buffer[1];     // Number of filters per feature
    fps->kernel_size_cols = buffer[2];    // Filter cols
    fps->kernel_size_rows = buffer[3];    // Filter rows
	
    unsigned total_filters_memory = sizeof(float)*(fps->in_channels * fps->out_channels * fps->kernel_size_cols * fps->kernel_size_rows);
    
    // Allocate memory
    checkCudaErrors(cudaMallocHost(&filters, total_filters_memory));
    checkCudaErrors(cudaMallocHost(&bias, sizeof(float)*fps->out_channels));
    
    if (bias == NULL){
    	fprintf(stderr, "Failed to allocate bias host matrix!\n");
    	exit(EXIT_FAILURE);
    }
    if (filters == NULL){
	fprintf(stderr, "Failed to allocate filters host matrix!\n");
	exit(EXIT_FAILURE);
    }
     
    // Fill bias matrix
    for(int k = 0; k < fps->out_channels; k++)
	bias[k] = buffer[k + DATA_OFFSET];


    unsigned bias_offset = fps->out_channels;
    unsigned nb_features = fps->in_channels * fps->out_channels;	
	
    // Order data in form of rows, cols, depth, filters per feature
    for(int i = 0; i < fps->kernel_size_rows; i++)
    	for(int j = 0; j < fps->kernel_size_cols; j++)
    		for(int k = 0; k < fps->out_channels; k++)
			for(int l = 0; l < fps->in_channels; l++)
				filters[i*fps->kernel_size_cols*nb_features + j*nb_features + l*fps->out_channels + k] = buffer[i*fps->kernel_size_rows*fps->out_channels*fps->in_channels + j*fps->out_channels*fps->in_channels + k*fps->in_channels + l + DATA_OFFSET + bias_offset];
				
    delete[] buffer;

    *out_filters = filters;
    *out_bias = bias;

}


/* load_norm
 *
 * Description: Loads the convolution batch normalization data from binary files.
 *		 The data is divided in 1 unsigned variable of header
 *		 followed by a variable number batch normalization data 
 *		(output channels*4 being 4 for |{mean, var, weight, bias}|). 
 *
 * Parameter:   
 *		- float** out_mean: Pointer to batch normalization mean (array of elements)
 *		- float** out_var: Pointer to batch normalization variance (array of elements)
 *		- float** out_weight: Pointer to batch normalization weight (array of elements)
 *		- float** out_bias: Pointer to batch normalization bias (array of elements)
 *		- const char* PATH: Path to input filter weights file 

 *
 * Returns:     The number of extracted elements 
 *
 * */
unsigned load_norm(float** out_mean, float** out_var, float** out_weight, float** out_bias, const char* PATH){

  std::ifstream binfile;
  binfile = std::ifstream(PATH, std::ios::in | std::ios::binary);

  unsigned length;
  float *buffer = NULL;
  
  float* mean;
  float* var;
  float* weight;
  float* bias;
  
  const unsigned DATA_OFFSET = 1;
  const unsigned NB_ELEMENTS = 4;
  unsigned num_features;

    if(binfile){
	    // get length of file:
	    binfile.seekg (0, binfile.end);
	    length = binfile.tellg() / sizeof(float);
	    binfile.seekg (0, binfile.beg);

	    buffer = new float[length];

	    // read data 
	    binfile.read(reinterpret_cast<char*>(buffer), sizeof(float)*length);

	    binfile.close();
    }else{
 	    fprintf(stderr, "Failed to locate the binary file!\n");
	    exit(EXIT_FAILURE);
    }
    
    
    num_features = buffer[0];      // Number of features, i.e., the nb of output channels of the conv weights file
    unsigned element_memory = sizeof(float)*num_features;
    
    // Allocate memory
    checkCudaErrors(cudaMallocHost(&mean, element_memory));
    checkCudaErrors(cudaMallocHost(&var, element_memory));
    checkCudaErrors(cudaMallocHost(&weight, element_memory));
    checkCudaErrors(cudaMallocHost(&bias, element_memory));
    
    
    if (mean == NULL){
    	fprintf(stderr, "Failed to allocate the mean host matrix!\n");
    	exit(EXIT_FAILURE);
    }
    if (var == NULL){
	fprintf(stderr, "Failed to allocate the variance host matrix!\n");
	exit(EXIT_FAILURE);
    }
    if (weight == NULL){
	fprintf(stderr, "Failed to allocate the weight host matrix!\n");
	exit(EXIT_FAILURE);
    }
    if (bias == NULL){
	fprintf(stderr, "Failed to allocate the bias host matrix!\n");
	exit(EXIT_FAILURE);
    }
     
    // Fill matrices
    for(int k = 0, i = 0; k < num_features; k++, i+=NB_ELEMENTS){
    	mean[k] = buffer[i + DATA_OFFSET + 0];	 // 0 offset for the mean
	var[k] = buffer[i + DATA_OFFSET + 1]; 	// 1 offset for the var
	weight[k] = buffer[i + DATA_OFFSET + 2];	// 2 offset for the weight
	bias[k] = buffer[i + DATA_OFFSET + 3];	// 3 offset for the bias
	
    }


    delete[] buffer;

    *out_mean= mean;
    *out_var = var;
    *out_weight = weight;
    *out_bias = bias;
    
    
    return num_features; // Size for each element mean, var, weight, bias
    
}


/* wino_3x3filter_2x2transform
 *
 * Description: Performs the Winograd transformation of a 3x3 filter weights for an output tile of 2x2. 
 *
 * Parameter:   
 *		- float** h_wino_filter: Pointer to the computed transformed filter weights (array of elements)
 *		- float* original_filter: Pointer to the base address of the original filter weights 
 *		- struct filter_prop_struct *fps: Structure where the filter specs are saved
 *
 *
 * Returns:     Nothing
 *
 * */
void wino_3x3filter_2x2transform(float** h_wino_filter, float* original_filter, struct filter_prop_struct fps){
	
	float* wino_filter;
	unsigned wino_filter_mem = fps.in_channels * fps.out_channels * 16 * sizeof(float);
	
	checkCudaErrors(cudaMallocHost(&wino_filter, wino_filter_mem));

  
	const float G[12] = {1, 0, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0, 0, 1};
	const float GT[12] = {1, 0.5, 0.5, 0, 0, 0.5, -0.5, 0, 0, 0.5, 0.5, 1};
	
	unsigned ch_off = fps.in_channels * fps.out_channels;
	
	for(int cnt = 0; cnt < ch_off; cnt++){

		float Gg[12] = {
		G[0]*original_filter[cnt + 0*ch_off] + G[1]*original_filter[cnt + 3*ch_off] + G[2]*original_filter[cnt + 6*ch_off],       
		G[0]*original_filter[cnt + 1*ch_off] + G[1]*original_filter[cnt + 4*ch_off] + G[2]*original_filter[cnt + 7*ch_off],   
		G[0]*original_filter[cnt + 2*ch_off] + G[1]*original_filter[cnt + 5*ch_off] + G[2]*original_filter[cnt + 8*ch_off], 
		
		G[3]*original_filter[cnt + 0*ch_off] + G[4]*original_filter[cnt + 3*ch_off] + G[5]*original_filter[cnt + 6*ch_off],       
		G[3]*original_filter[cnt + 1*ch_off] + G[4]*original_filter[cnt + 4*ch_off] + G[5]*original_filter[cnt + 7*ch_off],   
		G[3]*original_filter[cnt + 2*ch_off] + G[4]*original_filter[cnt + 5*ch_off] + G[5]*original_filter[cnt + 8*ch_off],  
		
		G[6]*original_filter[cnt + 0*ch_off] + G[7]*original_filter[cnt + 3*ch_off] + G[8]*original_filter[cnt + 6*ch_off],       
		G[6]*original_filter[cnt + 1*ch_off] + G[7]*original_filter[cnt + 4*ch_off] + G[8]*original_filter[cnt + 7*ch_off],   
		G[6]*original_filter[cnt + 2*ch_off] + G[7]*original_filter[cnt + 5*ch_off] + G[8]*original_filter[cnt + 8*ch_off], 
		
		G[9]*original_filter[cnt + 0*ch_off] + G[10]*original_filter[cnt + 3*ch_off] + G[11]*original_filter[cnt + 6*ch_off],       
		G[9]*original_filter[cnt + 1*ch_off] + G[10]*original_filter[cnt + 4*ch_off] + G[11]*original_filter[cnt + 7*ch_off],   
		G[9]*original_filter[cnt + 2*ch_off] + G[10]*original_filter[cnt + 5*ch_off] + G[11]*original_filter[cnt + 8*ch_off]     
		};
				
		float GgGT[16];
		GgGT[0] = Gg[0]*GT[0] + Gg[1]*GT[4] + Gg[2]*GT[8];  
		GgGT[1] = Gg[0]*GT[1] + Gg[1]*GT[5] + Gg[2]*GT[9];
		GgGT[2] = Gg[0]*GT[2] + Gg[1]*GT[6] + Gg[2]*GT[10]; 
		GgGT[3] = Gg[0]*GT[3] + Gg[1]*GT[7] + Gg[2]*GT[11]; 
		
		GgGT[4] = Gg[3]*GT[0] + Gg[4]*GT[4] + Gg[5]*GT[8];  
		GgGT[5] = Gg[3]*GT[1] + Gg[4]*GT[5] + Gg[5]*GT[9]; 
		GgGT[6] = Gg[3]*GT[2] + Gg[4]*GT[6] + Gg[5]*GT[10]; 
		GgGT[7] = Gg[3]*GT[3] + Gg[4]*GT[7] + Gg[5]*GT[11]; 
		  
		GgGT[8] = Gg[6]*GT[0] + Gg[7]*GT[4] + Gg[8]*GT[8];  
		GgGT[9] = Gg[6]*GT[1] + Gg[7]*GT[5] + Gg[8]*GT[9]; 
		GgGT[10] = Gg[6]*GT[2] + Gg[7]*GT[6] + Gg[8]*GT[10]; 
		GgGT[11] = Gg[6]*GT[3] + Gg[7]*GT[7] + Gg[8]*GT[11]; 
		 
		GgGT[12] = Gg[9]*GT[0] + Gg[10]*GT[4] + Gg[11]*GT[8];  
		GgGT[13] = Gg[9]*GT[1] + Gg[10]*GT[5] + Gg[11]*GT[9]; 
		GgGT[14] = Gg[9]*GT[2] + Gg[10]*GT[6] + Gg[11]*GT[10]; 
		GgGT[15] = Gg[9]*GT[3] + Gg[10]*GT[7] + Gg[11]*GT[11]; 
		
		
		wino_filter[cnt + 0*ch_off] = GgGT[0];  
		wino_filter[cnt + 1*ch_off] = GgGT[1];
		wino_filter[cnt + 2*ch_off] = GgGT[2]; 
		wino_filter[cnt + 3*ch_off] = GgGT[3]; 
		
		wino_filter[cnt + 4*ch_off] = GgGT[4];  
		wino_filter[cnt + 5*ch_off] = GgGT[5]; 
		wino_filter[cnt + 6*ch_off] = GgGT[6]; 
		wino_filter[cnt + 7*ch_off] = GgGT[7]; 
		  
		wino_filter[cnt + 8*ch_off] = GgGT[8];  
		wino_filter[cnt + 9*ch_off] = GgGT[9]; 
		wino_filter[cnt + 10*ch_off] = GgGT[10]; 
		wino_filter[cnt + 11*ch_off] = GgGT[11]; 
		 
		wino_filter[cnt + 12*ch_off] = GgGT[12];  
		wino_filter[cnt + 13*ch_off] = GgGT[13]; 
		wino_filter[cnt + 14*ch_off] = GgGT[14]; 
		wino_filter[cnt + 15*ch_off] = GgGT[15];   

	}
	
	*h_wino_filter = wino_filter;

}



