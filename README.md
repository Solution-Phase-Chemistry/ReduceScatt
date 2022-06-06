# ReduceDatav2
Reduce scattering data broken up into smaller functions

## There are three main dictionaries to run the code:  
### Input dictionaries:
paramDict:  see ParamDictRef.txt.txt file of possible parameters and explanations, these are processing parameters \
varDict:  dictionary of variables desired and used by code and where they are found in the .h5 file

### Output :
outDict:  see outDictRef.txt files for what parameters may be saved in this dictionary.  This dictionary is eventually saved as an .npy file if you save the data


## Notebooks 
ScatteringSteps.ipynb shows the individual subfunctions used to process the data \
ScatteringTest.ipynb tests the ReduceData function that calls the individual subfunctions in sequence. 

## LCLSDataToolsNew
 --Fns.py are the subfunctions used to process and reduce the data.  Use these subfunctions if you want to do the steps in sequence or to create a custom ReduceData function \
--Tools.py are sub-sub functions called from the subfunctions. 
