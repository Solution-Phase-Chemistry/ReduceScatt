# ReduceDatav2
Reduce scattering data broken up into smaller functions

## There are three main dictionaries to run the code:  
### Input dictionaries:
paramDict:  see ParamDictRef.txt.txt file of possible parameters and explanations, these are processing parameters \
varDict:  dictionary of variables desired and used by code and where they are found in the .h5 file

### Output :
outDict:  see outDictRef.txt files for what parameters may be saved in this dictionary.  This dictionary is eventually saved as an .npy file if you save the data


## .ipynb
ScatteringSteps.ipynb shows the individual subfunctions used to process the data \
ScatteringTest.ipynb tests the Reduce function that calls the individual subfunctions in sequence. 
