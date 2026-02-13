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




def StackProccessed(inpath,exper,runs,base=None, method='WAve'):
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
            AllErr.append(data1['diff_std'])
        
        
    ## check that all ts and qs are the same or throw error
    try: 
        AllTs=np.array(AllTs,dtype=float)
        ts=np.unique(AllTs,axis=0).squeeze()
        assert len(ts.shape)==1
    except:
        print('more than one unique t axis, use Stack_Interp instead')
        

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

    AllData=np.array(AllData) # runs x ts x phis x qs array
    AllBC=np.array(AllBC) #runs x ts 
    
    ##  weigh each run by number of shots per bin and sum, then divide by total shots in bin
    if method=='bincount':
        AllD2=divAny(AllData,1/AllBC,axis=(2,3,0,1)) ## multiply sig/shot and BC (shots/bin)
        sumD=np.nansum(AllD2,axis=0) ##total signal for each bin
        sumBC=np.nansum(AllBC,axis=0) ##total shots per bin
        aveD=divAny(sumD,sumBC,axis=(1,2,0)) #average signal per shot per bin
        aveD[np.nonzero(sumBC==0),:,:]=np.nan
        
        #standard error= sqrt((sum(bc_i*(x_i-<x>)**2))/(sum(bc_i-1)*sum(bc_i)))
        AllErr1=divAny((AllData-aveD)**2,1/AllBC,axis=(2,3,0,1))
        sumAE1=np.nansum(AllErr1,axis=0)
        ssBC=(sumBC-1)*(sumBC)
        AllErr2=divAny(sumAE1,ssBC,axis=(1,2,0))
        Derr=np.sqrt(AllErr2)
        Derr[np.nonzero(sumBC==0),:,:]=np.nan
    
        
        stackDict={'aveData':aveD,'errData':Derr,'sumBC':sumBC,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
        return stackDict
    
    if method=='WAve':
        AllErr=np.array(AllErr)
        aveD,Derr=WAve(AllData,AllErr,axis=0)
        stackDict={'aveData':aveD,'errData':Derr,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
        return stackDict
    
    if method=='Sum':
        sumD=np.nansum(AllData,axis=0)
        sumBC=np.nansum(AllBC,axis=0) ##total shots per bin
        stackDict={'sumData':sumD,'sumBC':sumBC,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
        return stackDict
    
    


def Stack_Interp(inpath,exper,runs,base=None,method='Ave'):
    ''' for runs in experiment, load .npy files from inpath and stack runs using method specified.
    Interpolates wrt time axis,  no bin_error/bin_count weights. 
    Methods that return average signal per t bin:
    'Ave'
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
    t_min=[]
    t_max=[]
    t_num=[]
    for ii,run in enumerate(runs):
        if base is None:
            data1=np.load(inpath+exper+'_Run%04i_out.npy'%run,allow_pickle=True).item()
        else:
            data1=np.load(inpath+exper+'_Run%04i'%run+base+'_out.npy',allow_pickle=True).item()
        
        temp_time=data1['xcenter']
        AllTs.append(temp_time)
        t_min.append(np.nanmin(temp_time))
        t_max.append(np.nanmax(temp_time))
        t_num.append(temp_time.shape[0])
        AllData.append(data1['diff_bin'])
        AllQs.append(data1['qs'])
        AllPhis.append(data1['phis'])
        AllBC.append(data1['xbin_occupancy'])

    # interpolate
    ts=np.linspace(np.nanmean(np.array(t_min)),np.nanmean(np.array(t_max)),np.nanmean(np.array(t_num)).astype(int))
    for ii, run in enumerate(runs):
        temp=scipy.interpolate.interp1d(AllTs[ii], AllData[ii], kind='linear', axis=0, 
                      copy=True, bounds_error=False, fill_value=np.nan, assume_sorted=False)
        AllData[ii]=temp(ts)

        
        
    ## check that all qs and phis are the same or throw error
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

    AllData=np.array(AllData) # runs x ts x phis x qs array
    
    
    if method=='Ave':
        aveD=np.nanmean(AllData,axis=0)
        stackDict={'aveData':aveD,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
        return stackDict
    
    if method=='Sum':
        sumD=np.nansum(AllData,axis=0)
        stackDict={'sumData':sumD,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
        return stackDict

    
    
    
def Stack_eData(inpath,exper,runs,earlyt=None,base=None):
    '''stack data with weighted average,  
    calculate standard deviation for early time range (pre time zero) for each run, the total stack, and for the stack as you build it'''
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
        AllErr.append(data1['diff_std'])


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

    try:
        AllPhis=np.array(AllPhis,dtype=float)
        phis=np.unique(AllPhis,axis=0).squeeze()
        assert len(phis.shape)==1
    except:
        print('more than one unique phi axis')

    AllData=np.array(AllData) # runs x ts x phis x qs array
    AllBC=np.array(AllBC) #runs x ts 

    ## do weighted average
    AllErr=np.array(AllErr)
    aveD,Derr=WAve(AllData,AllErr,axis=0)
    
    stackDict={'aveData':aveD,'errData':Derr,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':'WAve'}
    
    #calculate standard deviations early time for each run
    if earlyt is not None:
        eet,earlyt=chooseR(earlyt[0],earlyt[1],ts)
        AlleSTD=np.nanstd(AllData[:,eet,:,:],axis=(1,2,3))

        #calc stdev early time for stack
        eSTD_stack=np.nanstd(aveD[eet,:,:])

        #calc stdev early time as build up stack from one run
        All_estd_build=np.zeros(AllData.shape[0])
        for ii in range(AllData.shape[0]):
            temp_ave,temp_err=WAve(AllData[:ii+1],AllErr[:ii+1],axis=0)
            All_estd_build[ii]=np.nanstd(temp_ave[eet,:,:])

        estat={'AlleSTD':AlleSTD,'eSTD_stack':eSTD_stack,'All_estd_build':All_estd_build}
        
    else:
        estat=None
    
        
    return stackDict,estat









def eData_plot(runs,estat,plot_vs_run=False):
    ''' plot early time standard deviation data for stack.
        if plot_vs_run is true then plot vs run number, otherwise plot vs number of scans'''
    
    if plot_vs_run:
        xx=runs
        xlabel='run number'
    else:
        xx=range(len(runs))
        xlabel='number of runs'
    
    plt.figure('eet')
    plt.subplot(2,1,1)
    plt.plot(xx,estat['AlleSTD'],color='b',marker='s',ls='')
    # plt.axhline(eSTD_stack,color='r',ls='--',label='early time stdev for stack')
    plt.title('early time stdev per run')
    # plt.legend()
    plt.tick_params('x', labelbottom=False)

    plt.subplot(2,1,2)
    plt.plot(xx,estat['All_estd_build'],'k.')
    plt.title('early time stdev stacking up to run')
    plt.xlabel(xlabel)
    plt.show()




### cosine similarity included ###
def similarity(A,B):
    if A.shape != B.shape:
        print('Warning different sized arrays!')
        return
        
    A1=A.flatten()
    B1=B.flatten()
    dot=np.nansum(A1*B1)
    A1_norm=np.sqrt(np.nansum(A1*A1))
    B1_norm=np.sqrt(np.nansum(B1*B1))
                    
    return dot/(A1_norm*B1_norm)

def Stack_stats(inpath,exper,runs,earlyt=None,base=None):
    '''stack data with weighted average,  
    calculate standard deviation for early time range (pre time zero) for each run, 
    the total stack, and for the stack as you build it
    also calculate cosine similarity for each run wrt stack so far and wrt total stack'''
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
        AllErr.append(data1['diff_std'])
    
    
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
    
    try:
        AllPhis=np.array(AllPhis,dtype=float)
        phis=np.unique(AllPhis,axis=0).squeeze()
        assert len(phis.shape)==1
    except:
        print('more than one unique phi axis')
    
    AllData=np.array(AllData) # runs x ts x phis x qs array
    AllBC=np.array(AllBC) #runs x ts 
    
    ## do weighted average
    AllErr=np.array(AllErr)
    aveD,Derr=WAve(AllData,AllErr,axis=0)
    
    stackDict={'aveData':aveD,'errData':Derr,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':'WAve'}
    
    if earlyt is not None:
        eet,earlyt=chooseR(earlyt[0],earlyt[1],ts)
        AlleSTD=np.nanstd(AllData[:,eet,:,:],axis=(1,2,3))
    
        #calc stdev early time for stack
        eSTD_stack=np.nanstd(aveD[eet,:,:])
    
        #set up for loop
        All_estd_build=np.zeros(AllData.shape[0])
    
    #set up 
    All_sim_stack=np.zeros(AllData.shape[0])
    All_sim_build=np.zeros(AllData.shape[0])
    #calc stdev early time and similarity as build up stack from one run
    for ii in range(AllData.shape[0]):
        temp_ave,temp_err=WAve(AllData[:ii+1],AllErr[:ii+1],axis=0)
    
        #similarity wrt full stack
        All_sim_stack[ii]=similarity(AllData[ii],aveD)
        All_sim_build[ii]=similarity(AllData[ii],temp_ave)
        
        
        try:
            All_estd_build[ii]=np.nanstd(temp_ave[eet,:,:])
        except:
            continue
    simStat={'All_sim_build':All_sim_build,'All_sim_stack':All_sim_stack}
    try:
        estat={'AlleSTD':AlleSTD,'eSTD_stack':eSTD_stack,'All_estd_build':All_estd_build}
    except:
        estat=None
    
    return stackDict,simStat,estat

        
    
def sim_plot(runs,simStat,plot_vs_run=False):
    ''' plot similarity measures after running Stack_stats
        if plot_vs_run is true then plot vs run number, otherwise plot vs number of scans'''
    
    if plot_vs_run:
        xx=runs
        xlabel='run number'
    else:
        xx=range(len(runs))
        xlabel='number of runs'
        
    plt.figure('sim')
    plt.subplot(2,1,1)
    plt.plot(xx,simStat['All_sim_build'],color='r',marker='^')
    # plt.axhline(eSTD_stack,color='r',ls='--',label='early time stdev for stack')
    plt.title('cosine similarity as build stack')
    # plt.legend()
    plt.tick_params('x', labelbottom=False)

    plt.subplot(2,1,2)
    plt.plot(xx,simStat['All_sim_stack'],'k.')
    plt.title('cosine similarity with respect to complete stack')
    plt.xlabel(xlabel)
    plt.show()


