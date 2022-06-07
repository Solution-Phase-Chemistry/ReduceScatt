import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import h5py
import sys
import os
import time
import scipy
import scipy.io as scio

from LCLSDataToolsNew.GeneralTools import *
from LCLSDataToolsNew.filterTools import *
from LCLSDataToolsNew.diffSignalTools import *
from LCLSDataToolsNew.plottingTools import *
from LCLSDataToolsNew.binningToolsErr import *
from LCLSDataToolsNew.anisotropyToolsAll import *
from LCLSDataToolsNew.SVDTools import *

from sklearn.linear_model import RANSACRegressor as RSC
from sklearn.linear_model import LinearRegression as LR

from LCLSDataToolsNew.SetUpFns import *
from LCLSDataToolsNew.DiffBinFns import *
# from LCLSDataToolsNew.ReduceFns import *




def doAnisotropy(paramDict,outDict):
    print('start anisotropy')
    shift_n=paramDict['shift_n']
    ddata=outDict['diff_bin']
    try:
        qs=outDict['h5Dict']['qs']
        phis=outDict['h5Dict']['phis']
    except:
        qs=outDict['qs']
        phis=outDict['phis']
    
    S0, err_S0, S2, err_S2=S0S2(ddata,phis,fil=None,shift_n=shift_n,deg=None)
    outDict.update({'S0':S0,'S0_err':err_S0,'S2':S2,'S2_err':err_S2})
        

        

    
    
def saveDictionary(outpath,paramDict,outDict):
    '''delete h5Dict from outDict, save outDict to .npy file in directory=outpath
    '''
    
    #add things we want
    outDict['qs']=outDict['h5Dict']['qs']
    outDict['phis']=outDict['h5Dict']['phis']
    outDict['paramDict']=paramDict
    
    #delete things we don't want to save
    removeL=['h5Dict','x_Data', 'diff_Data','xs','I0']
    [outDict.pop(key,None) for key in removeL]
    
    
    fout=outpath+outDict['h5name']+'_out.npy'
    np.save(fout,outDict)
    print('saved output to ', fout)
    
    
    
    
    
def ReduceData(inDir,exper,runs,outDir,paramDict,varDict):
    for run in runs:
        fname=inDir+exper+'_Run%04i.h5'%run
        print('loading ', fname)
        outDict={}
        then=time.time()
        LoadH5(fname,varDict,paramDict, outDict)
        setupFilters(paramDict,outDict)
        I0Filters(paramDict,outDict)
        eBeamFilter(paramDict,outDict)
        TTfilter(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
        MakeScanAx(paramDict,outDict,tt_corrNew=None)
        DarkSubtract(paramDict,outDict)
        EnergyCorr(paramDict,outDict)
        DetectorNonlinCorr(paramDict,outDict)
        NormalFactor(paramDict,outDict)
        doDifference(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
    
        doTimeBinning(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
        if paramDict['aniso']:
            doAnisotropy(paramDict,outDict)
        saveDictionary(outDir,paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
        
        
        
        
        
        
def StackProccessed(inpath,exper,runs,method='bincount'):
    ''' for runs in experiment, load .npy files from inpath and stack runs using method specified.
    Methods that return average signal per t bin:
    'bincount' = weigh each run by number of shots per bin and sum, then divide by total shots in bin
    'WAve' = weighted average for each bin using bin_err
    Methods that return total signal per t bin:
    'Sum' = just sum values for each t bin
    '''
    
    ## load data
    AllData=[]
    AllTs=[]
    AllQs=[]
    AllBC=[]
    AllErr=[]
    for run in runs:
        with np.load(inpath+exper+'_Run%04i.npy'%run,allow_pickle=True).item() as data1:
            AllData.append(data1['diff'])
            AllTs.append(data1['ts'])
            AllQs.append(data1['qs'])
            AllBC.append(data1['bincount'])
            if method=='WAve':
                AllErr.append(data1['diff_err'])
        
        
    ## check that all ts and qs are the same or throw error
    try: 
        AllTs=np.array(AllTs,dtype=float)
        ts=np.unique(AllTs,axis=0).squeeze()
        assert len(ts.shape)==1
    except:
        print('more than one unique t axis')

    try:
        AllQs=np.array(AllQs,dtype=float)
        qs=np.unique(AllQs,axis=0).squeeze()
        assert len(qs.shape)==1
    except:
        print('more than one unique q axis')
        
    AllData=np.array(AllData) # runs x ts x phis x qs array
    AllBC=np.array(AllBC) #runs x ts 
    
    ##  weigh each run by number of shots per bin and sum, then divide by total shots in bin
    if method=='bincount':
        AllD2=divAny(AllData,1/AllBC) ## multiply sig/shot and BC (shots/bin)
        sumD=np.nansum(AllD2,axis=0) ##total signal for each bin
        sumBC=np.nansum(AllBC,axis=0) ##total shots per bin
        aveD=divAny(sumD,sumBC) #average signal per shot per bin
        aveD[np.nonzero(sumBC==0),:,:]=np.nan
        stackDict={'aveData':aveD,'sumBC':sumBC,'ts':ts,'qs':qs,'runs':runs,'method':method}
        return stackDict
    
    if method=='WAve':
        AllErr=np.array(AllErr)
        aveD,Derr=WAve(AllData,AllErr,axis=0)
        stackDict={'aveData':aveD,'errData':Derr,'ts':ts,'qs':qs,'runs':runs,'method':method}
        return stackDict
    
    if method=='Sum':
        sumD=np.nansum(AllData,axis=0)
        sumBC=np.nansum(AllBC,axis=0) ##total shots per bin
        stackDict={'sumData':sumD,'sumBC':sumBC,'ts':ts,'qs':qs,'runs':runs,'method':method}
        return stackDict
        
    
    
    
        
        
    
    
    
        