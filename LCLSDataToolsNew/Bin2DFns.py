#### Sumana Raj######

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import h5py
import sys
import os
import time
import scipy
import scipy.io as scio
from scipy.stats import binned_statistic_dd


from LCLSDataToolsNew.GeneralTools import *
from LCLSDataToolsNew.filterTools import *
from LCLSDataToolsNew.diffSignalTools import *
from LCLSDataToolsNew.plottingTools import *
from LCLSDataToolsNew.binningToolsErr import *
from LCLSDataToolsNew.anisotropyToolsAll import *
from LCLSDataToolsNew.SVDTools import *

from LCLSDataToolsNew.SetUpFns import *
from LCLSDataToolsNew.DiffBinFns import *
from LCLSDataToolsNew.ReduceFns import *

def Bin2D(paramDict,outDict):
    ''' sets up time axis as dictated by binSetup and binSet2, sets up unique other scan axis.
    bins 2d '''
    ## inputs
    x1Data=outDict['x_Data']
    onData=outDict['diff_Data']
    scan2=outDict['h5Dict']['scan_vec2']
    f_intens=outDict['filters']['f_good']
    f_lon=outDict['filters']['f_lon']
    binSet=paramDict['binSetup']
    binSet2=paramDict['binSet2']
    
    x2Data=scan2[f_intens & f_lon]
    
    
          
###### set up 2D axes, calculate bin edges ########

    ### set up time axis 
    if binSet=='fixed':
        x1edges=binSet2
        x1centres=x1edges[0:-1]+np.diff(x1edges)/2
    elif binSet=='points':
        n=binSet2
        if n>0:
            x1edges,x1centres=MakeEqualBins(x1Data,n)
        else:
            fill,edges=np.histogram(x1Data,'auto')
            n=len(x1Data)/len(edges) #make that many bins, but do it by points. 
            x1edges,x1centres=MakeEqualBins(x1Data,n)
    elif binSet=='nbins':
        n=binSet2
        if n>0:
            fill,x1edges=np.histogram(x1Data,n)
        else:
            fill,x1edges=np.histogram(x1Data,'auto')
        x1centres=x1edges[0:-1]+np.diff(x1edges)/2
        
        
    ### set up other axis - always unique
    x2centres = np.unique(x2Data)

    shift=np.append(np.diff(x2centres)[0]/2,np.diff(x2centres)/2)
    x2edges=x2centres-shift
    x2edges=np.append(x2edges,x2centres[-1]+np.diff(x2centres)[-1])



    ######### bin ##########
    diff_bin=np.full((x1centres.shape[0],x2centres.shape[0],
                      onData.shape[1],onData.shape[2]),np.nan)
    
    
    for ii in range(onData.shape[1]):
        temp_in=onData[:,ii,:].T
        temp_out=binned_statistic_dd([x1Data, x2Data], temp_in, 
                                     statistic='mean', bins=[x1edges,x2edges],
                                     expand_binnumbers=True)
        diff_bin[:,:,ii,:]=temp_out[0].transpose(1,2,0)
    outDict.update({'x1centres':x1centres,'x2centres':x2centres,'diff_bin':diff_bin})
    print(outDict.keys())
    
    
    
    
    
def ReduceData2D(inDir,exper,runs,outDir,paramDict,varDict):
    for run in runs:
        paramDict['scan_var']='newdelay'
        varDict['scan_vec']='enc/lasDelay'
        varDict['scan_vec2']='scan/var0'
        
        fname=inDir+exper+'_Run%04i.h5'%run
        print('loading ', fname)
        outDict={}
        then=time.time()
        LoadH5(fname,varDict,paramDict, outDict)
        setupFilters(paramDict,outDict)
        I0Filters(paramDict,outDict)
        # eBeamFilter(paramDict,outDict)
        if paramDict['use_TT']==True or paramDict['use_TT']=='filter':
            TTfilter(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        saveReduction(outDir,outDict)
        
        
        MakeScanAx(paramDict,outDict,tt_corrNew=None)
        DarkSubtract(paramDict,outDict)
        if paramDict['energy_corr']:
            EnergyCorr(paramDict,outDict)
        if paramDict['NonLin_corr'] is not None:
            DetectorNonlinCorr(paramDict,outDict)
        NormalFactor(paramDict,outDict)
        doDifference(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
            
            
        
        Bin2D(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
        
        
        # if paramDict['aniso']:
        #     doAnisotropy(paramDict,outDict)
        saveDictionary(outDir+'npy/',paramDict,outDict)
        # overviewPlot(outDir+'figures/',paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")