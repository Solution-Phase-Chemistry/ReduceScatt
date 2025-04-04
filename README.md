# ReduceScatt (formerly ReduceDatav2)
Filter and time bin the azimuthally averaged X-ray scattering smalldata from LCLS.  

## There are three main dictionaries to run the code:  
### Input dictionaries:
**paramDict**:  see paramDict_details.md file of possible parameters and explanations, these are processing parameters \
**varDict**:  dictionary of variables desired and used by code and where they are found in the .h5 file.  See ScattCodeInfo.md for example 

### Output :
**outDict**:  see ScattCodeInfo.md files for what parameters may be saved in this dictionary.  This dictionary is eventually saved as an .npy file if you save the data


## Notebooks 
- Master_XSS: reduction and useful jupyter notebook cells for fitting data and stacking; this notebook is not intended to be run sequentially, but to have all the useful sections that you may choose to run
- example_overlap:  simple notebook that can be run in sequence for a lensh scan
- example_delay_v1: simple notebook that can be run in sequence for a set of delay scans
- StepbyStep:  Splits up the ReduceData function into its component steps, run them one at a time. 
### old notebooks 
- ScatteringSteps.ipynb shows the individual subfunctions used to process the data \
- ScatteringTest.ipynb tests the ReduceData function that calls the individual subfunctions in sequence. 
- XSS_noScanVar: example of how to reduce data with no scan variable


## LCLSDataToolsNew
 --Fns.py are the subfunctions used to process and reduce the data.  Use these subfunctions if you want to do the steps in sequence or to create a custom ReduceData function \
--Tools.py are sub-sub functions called from the subfunctions. \



