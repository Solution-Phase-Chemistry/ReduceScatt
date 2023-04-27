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



def RedNoScanV(inDir,exper,runs,outDir,paramDict1,varDict):
    ''' averages all shots in run after applying filters and corrections'''
    for run in runs:
        fname=inDir+exper+'_Run%04i.h5'%run
        paramDict=paramDict1.copy()
        print('loading ', fname)
        outDict={}
        then=time.time()
        LoadH5(fname,outDir,varDict,paramDict, outDict)
        setupFilters(paramDict,outDict)
        IscatFilters(paramDict,outDict)
        # eBeamFilter(paramDict,outDict)
        if paramDict['use_TT']==True or paramDict['use_TT']=='filter':
            TTfilter(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        saveReduction(outDir,paramDict,outDict)

        outDict['xs']=np.arange(outDict['h5Dict']['azav'].shape[0])
        # MakeScanAx(paramDict,outDict,tt_corrNew=None)
        DarkSubtract(paramDict,outDict)
        NormalFactor(paramDict,outDict)
        if paramDict['energy_corr']:
            EnergyCorr(paramDict,outDict)
        if paramDict['NonLin_corr'] is not None:
            DetectorNonlinCorr(paramDict,outDict)
        doDifference(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")

        AveAllShots(paramDict, outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")

        if paramDict['aniso']:
            doAnisotropy(paramDict,outDict)
        saveDictionary(outDir+'npy/',paramDict,outDict)
        overviewPlot(outDir+'figures/',paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
        
        
        
        
        

        
        
def StackNoScanVar(inpath,exper,runs,base=None, method='bincount'):
    ''' for runs in experiment, load .npy files from inpath and stack runs using method specified.
    Methods that return average signal per t bin:
    'bincount' = weigh each run by number of shots per bin and sum, then divide by total shots in bin
    'WAve' = weighted average for each bin using bin_err
    Methods that return total signal per t bin:
    'Sum' = just sum values for each t bin
    
    base=None or base='_01' for inpath+exper+'_Run%04i_01_out.npy' etc 
    '''
    
    ## load data
    AllData=[]
    AllTs=[]
    AllQs=[]
    AllPhis=[]
    AllBC=[]
    AllErr=[]
    for run in runs:
        if base is None:
            data1=np.load(inpath+exper+'_Run%04i_out.npy'%run,allow_pickle=True).item()
        else:
            data1=np.load(inpath+exper+'_Run%04i'%run+base+'_out.npy',allow_pickle=True).item()
        AllData.append(data1['diff_bin'])
        AllTs.append(data1['xcenter'])
        AllQs.append(data1['qs'])
        AllPhis.append(data1['phis'])
        AllBC.append(data1['xbin_occupancy'])
        if method=='WAve':
            AllErr.append(data1['diff_err'])
        
        
    ## check that all ts and qs are the same or throw error 
    AllTs=np.array(AllTs,dtype=float)
    ts=np.array([0])

    try:
        AllQs=np.array(AllQs,dtype=float)
        qs=np.unique(AllQs,axis=0).squeeze()
        assert len(qs.shape)==1
    except:
        print('more than one unique q axis')
        
    try:
        AllPhis=np.array(AllPhis,dtype=float)
        phis=np.unique(AllPhis,axis=0).squeeze()
        assert len(phis.shape)==1
    except:
        print('more than one unique phi axis')

    AllData=np.array(AllData).squeeze() # runs x phis x qs array
    AllBC=np.array(AllBC) #runs 
    ##  weigh each run by number of shots per bin and sum, then divide by total shots in bin
    if method=='bincount':
        AllD2=divAny(AllData,1/AllBC,axis=(1,2,0)) ## multiply sig/shot and BC (shots/bin)
        sumD=np.nansum(AllD2,axis=0) ##total signal for each bin
        sumBC=np.nansum(AllBC,axis=0) ##total shots per bin
        aveD=divAny(sumD,sumBC,axis=(1,0)) #average signal per shot per bin
        stackDict={'aveData':aveD,'sumBC':sumBC,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
        return stackDict
