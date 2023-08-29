This read me helps you to set up an environment to run and train the segnet pytorch models for the UDD and UAVID datasets.

------------------------------------
 Step 1: Download datasets
------------------------------------
- Download the UDD-6 or UAVID dataset within this folder
- UAVID: https://uavid.nl/
- UDD: https://github.com/MarcWong/UDD

------------------------------------
 Step 2: Verify dataset
------------------------------------
- Rename the downloaded folders and verify folders organisation
- UAVID must be:  uavid 
					|-----> uavid_test
					|-----> uavid_train
					|-----> uavid_val
					
- UDD6 must be:  UDD6 
					|-----> metada
					|-----> train
					|-----> val
					
------------------------------------
 Step 3: Scale the dataset
------------------------------------
- Open the python file "scale_udd.py" or "scale_uavid.py"
- Choose your scaled size (default: X, Y = 360, 480)
- Run the script

------------------------------------
 Step 4: Execute training script
------------------------------------
- Open the python file "run_udd_segnet_X.py" or "run_uavid_segnet_X.py"
- Select the segnet to run (by commenting ou tthe others, per default this is segnet 1, the original with cl_segnet_1)
- You can change also some parameters such as the epochs (default is 300) etc...
- Run it! 
- It will generate the pth file for the trained model and associated csv file with all the curves.


Contacts:
# jean-baptiste.chaudron@isae-supaero.fr



