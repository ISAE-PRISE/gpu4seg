# GPU4SEG project

Copyright (C) 2023 ISAE-SUPAERO

The objective of this project is to study the response of image Segmentation CNN models to architectural (e.g. n° of levels, layer or channels) and implementation (e.g. GPU utilization, convolution execution) modifications.
Please note that the code and data publicly released in this repository is a subset of a major production.  

## Description

1. "pytorch/": This workspace contains the Pytorch envrionment to train the different Segnet architectures.

2. "cuda/segnet/": This workspace contains the CUDA code for executing different Segnet architectures making use of traditional convolutions and 2x2 output tiles Winograd convolutions.
						  
						  
## Hardware/Software Requirements

1. NVIDIA GPU. Tests were conducted using CUDA 11.4, and Jetson AGX Orin 64 GB and Jetson Xavier NX as MPSoCs.
   
2. stb libraries for importing and exporting images. Git: https://github.com/nothings/stb

3. Datasets. UAVID and UDD were used. The released testing code is configured for UAVID. For this dataset, the static and moving cars were assumed to belong to the same class. 

4. Binary files with the filter weights and normalization values. The training was performed using Pytorch.

## Binary files (pre-trained models)

The binary files for the pre-trained model are available on the ISAE-SUPAERO Dataverse for the PRISE project. There are two archives (UDD6 and UAVID datasets) which contain the binary files for Segnet architectures 1, 2, 3, 4, 6, 8 and 9.

UAVID : https://doi.org/10.34849/2K7NOZ

UDD6  : https://doi.org/10.34849/6EOIV0


## Authors

Jean-Baptiste Chaudron (ISAE-SUPAERO): Responsible of the Segnet model design and pytorch training.

Alfonso Mascarenas-Gonzalez (ISAE-SUPAERO): Responsible of the iGPU implementation.


## Contacts

jean-baptiste.chaudron@isae-supaero.fr

alfonso.mascarenas-gonzalez@isae-supaero.fr

## License

This project is distributed as an open-source software under GPL version 3.0.

## BibTeX Citation

Not available yet. Conference paper accepted and waiting for publication. 
