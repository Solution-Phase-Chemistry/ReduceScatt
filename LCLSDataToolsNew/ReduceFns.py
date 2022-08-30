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
        

        
        
        
        
        
        
def doSVDBackSub(paramDict,outDict,earlytrange=(-.5e-12, 0)):
    ##### do SVD back (t<0) subtraction #######    
    ts=outDict['xcenter']
    qrange=paramDict['qrange']
    diff_temp=outDict['diff_bin']
    
    goodq = np.where((qs > qrange[0]) & (qs < qrange[1]))[0]
    earlyt = np.where((ts > earlytrange[0]) & (ts < earlytrange[1]))[0]  
    data2d = np.nanmean(diff_temp,1)
    data2dgq = data2d[:, goodq]
    
    spectra, singVal, timetrace = do_svd(qs[goodq], ts[earlyt], data2dgq[earlyt,:], n = 10)

    N = 1  #number of svd components to subtract
    #weights = singVal[:N]
    Spectra = spectra
    #Data1 = data2dgq
    timeWeight = np.nanmean(timetrace[:,0])
    scale = np.zeros_like(Spectra[0])
    for x in range(N):
        timeWeight = np.nanmean(timetrace[:,x])
        scale += singVal[x]*Spectra[x]*timeWeight
    Bsub3d=diff_temp[:,:,goodq]-scale[None,None,:]  

######## calculate mean and var at low q with and without subtraction #########
#     #lowQRange=(.6,3)
#     lowQs1=np.where((qs > 0.6) & (qs < 3)[0])
#     lowQs2 = np.where((qs[goodq] > 0.6) & (qs[goodq] < 3)[0])

#     data2dlowQ = data2d[:, lowQs1].squeeze()
#     originalMean = np.nanmean(data2dlowQ[earlyt, :], (0,1))
#     originalVar = np.var(data2dlowQ[earlyt, :],(0,1))

#     BsublowQ = np.nanmean(Bsub3d[:,:, lowQs2],1).squeeze()
#     BsubMean = np.nanmean(BsublowQ[earlyt, :], (0,1))
#     BsubVar = np.var(BsublowQ[earlyt, :],(0,1))

#     print('Run',basename)
#     print("Uncorrected t<0, q=(0.6,3) mean: %0.3e var: %0.3e" %(originalMean, originalVar))
#     print("Corrected  t<0, q=(0.6,3) mean: %0.3e var: %0.3e" %(BsubMean,BsubVar))                  

    diff_temp[:,:,goodq] = Bsub3d
    outDict['diff_bin']=diff_temp
        
        
        
        
        

    
    
def saveDictionary(outpath,paramDict,outDict):
    '''delete h5Dict from outDict, save outDict to .npy file in directory=outpath
    '''
    
    #add things we want
    outDict['qs']=outDict['h5Dict']['qs']
    outDict['phis']=outDict['h5Dict']['phis']
    outDict['paramDict']=paramDict
    outDict['numshots_used']=outDict['x_Data'].shape[0]
    
    #delete things we don't want to save
    removeL=['h5Dict','x_Data', 'diff_Data','xs','Iscat']
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
        IscatFilters(paramDict,outDict)
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
    
        doTimeBinning(paramDict,outDict)
        now = time.time() #Time after it finished
        print(now-then, " seconds")
        
        if paramDict['aniso']:
            doAnisotropy(paramDict,outDict)
        saveDictionary(outDir+'npy/',paramDict,outDict)
        overviewPlot(outDir+'figures/',paramDict,outDict)
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
    AllPhis=[]
    AllBC=[]
    AllErr=[]
    for run in runs:
        data1=np.load(inpath+exper+'_Run%04i_out.npy'%run,allow_pickle=True).item()
        AllData.append(data1['diff_bin'])
        AllTs.append(data1['xcenter'])
        AllQs.append(data1['qs'])
        AllPhis.append(data1['phis'])
        AllBC.append(data1['xbin_occupancy'])
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
        AllD2=divAny(AllData,1/AllBC) ## multiply sig/shot and BC (shots/bin)
        sumD=np.nansum(AllD2,axis=0) ##total signal for each bin
        sumBC=np.nansum(AllBC,axis=0) ##total shots per bin
        aveD=divAny(sumD,sumBC) #average signal per shot per bin
        aveD[np.nonzero(sumBC==0),:,:]=np.nan
        stackDict={'aveData':aveD,'sumBC':sumBC,'ts':ts,'qs':qs,'phis':phis,'runs':runs,'method':method}
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
        
    
    
    
        
        
    
def overviewPlot(figdir,paramDict,outDict):
    ''' '''
    ## inputs
    diff=outDict['diff_bin']
    phis=outDict['phis']
    ts=outDict['xcenter']
    qs=outDict['qs'] 
    cake=outDict['loff_cake']
    scanvar=paramDict['scan_var']
    numshots=outDict['numshots']
    numshots_used=outDict['numshots_used']
    basename=outDict['h5name']
    pts_per_bin = outDict['xbin_occupancy']
    
    aniso=paramDict['aniso']
    enforce_iso=paramDict['enforce_iso']
    qrange=paramDict['qrange']
    show_svd=paramDict['showSVD']
    svd_n=paramDict['SVD_n']
    slice_plot=paramDict['slice_plot']
    smooth=paramDict['smooth']
    
    if qrange is None:
        qroi=np.arange(len(qs))
    else:
        qroi = np.where((qs > qrange[0]) & (qs < qrange[1]))[0]
        #qroi = np.arange(23,422)
        #print(qroi)   
        
    if enforce_iso: #if we are doing the iso correction:
        diff=diff/outDict['iso_corr']
        cake=cake/outDict['iso_corr']
    
    resfig=plt.figure('res')#results figure
    if aniso: #how many rows of plots will we need?
        nplot=4
    else:
        nplot=2
    resfig.clf()
    resfig.set_size_inches(9, nplot*3)
    if show_svd:
        svdfig=plt.figure('svd')
        svdfig.clf()
        svdfig.set_size_inches(9,(nplot-1)*3)
     
    ####plot cake 2d
    try:
        plot_2d(phis,qs,cake.T,fig='res',sub='%i21'%nplot,cb=False)
    except:
        plot_2d(phis[:-1],qs,cake.T,fig='res',sub='%i21'%nplot,cb=False)
    if enforce_iso:
        plt.ylabel('Q | slices, iso enforced')
    else:
        plt.ylabel('Q | azav slices')
    plt.xlabel('phi')
    
    ##### plot the average of each slice; should be flat
    ax2=plt.twinx()
    avg=np.nanmean(cake,1)
    ax2.plot(phis[:-1]+0.5*np.diff(phis),avg,'-o',color='w')
    ax2.set_ylim([0,np.nanmax(avg)])
    
    ##### plot the 1d average curve of the same
    plt.figure('res')
    plt.subplot(nplot,2,2)
    plt.plot(qs,np.nanmean(cake,0))
    plt.xlabel('Q')
    plt.ylabel('azav')

    print('plotting azavs')
    #xData=x[f_intens&f_lon]


    logscan=(np.abs(np.nanmax(ts)/np.nanmin(ts)))>1e3 # is the range we are scanning a lot of orders of magnitude? If so, plot nicer
    print('logscan '+str(logscan))
    everyother=(np.arange(len(ts))%2==0)
    diff2d=np.nanmean(diff,1)#average over phis
    if len(diff.shape)==2:
        diff2d=diff

    if scanvar != 'newdelay':
        x_lab=scanvar
    else:
        x_lab='t (s)'
        
        
     
    ####plot difference signal Q vs T heat map
    ax3 = plt.subplot(nplot,2,3)
    print(diff2d.shape)
    print(qs.shape)
    plot_2d(ts,qs[qroi],diff2d[:,qroi],fig='res',
            sub='%i23'%nplot,cb=False,logscan=logscan)
    plt.ylabel('Q | with TT')
    plt.xlabel(x_lab)
    

                     

    ####plot time slices DiffSignal vs Q
    
    if slice_plot is None:
        #plot every other time slice, DiffSig vs Q
        diff2d_bow=diff2d[everyother]
        plot_bow(qs[qroi],diff2d_bow[:,qroi],fig='res',sub=(nplot,2,4))
    else:
        plt.figure('res')
        slax=plt.subplot((nplot,2,4),sharex=ax3)
        plot_slice(ts,qs,diff2d,slice_plot,ax=slax,logscan=logscan)
        #plt.plot(ts,np.nanmean(diff2d[:,slice_plot],1),'o')
        plt.ylabel('Signal at q=%s-%s'%(qs[slice_plot[0]],qs[slice_plot[-1]]))
        plt.xlabel(x_lab)   
    
    
    if aniso:
        print('plottingn aniso')
        print(diff.shape)
        print(phis.shape)

        S0 = outDict['S0']
        S2=outDict['S2']
        
        S0_bow = S0[everyother]
        plot_bow(qs[qroi],S0_bow[:,qroi],fig='res',sub=(4,2,6))
        plt.ylabel('S0')
        
        S2_bow = S2[everyother]
        plot_bow(qs[qroi],S2_bow[:,qroi],fig='res',sub=(4,2,8))
        plt.ylabel('S2')
        
        ax5 = plt.subplot(4,2,5,sharex=ax3,sharey=ax3)
        plot_2d(ts,qs[qroi],S0[:,qroi],fig='res',sub=(4,2,5),cb=False)
        plt.ylabel('Q | S0')
        plt.xlabel(x_lab)
        ax7 = plt.subplot(4,2,7,sharex=ax3,sharey=ax3)
        plot_2d(ts,qs[qroi],S2[:, qroi],fig='res',sub=(4,2,7),cb=False)
        plt.ylabel('Q | S2')
        plt.xlabel(x_lab)
    

    
    ######## do SVDs #######

    if show_svd:
            plt.figure('svd')
            svdfig.add_subplot(3,3,1)
            plt.ylabel('azav')
            a=do_svd_protected(qs[qroi],ts,diff2d[:,qroi],n=svd_n,smooth=smooth,
                               logscan=logscan,fig='svd',sub=(nplot-1,3,1))
    
    if show_svd and aniso:
        #if at any point S0S2 failed to fit and returned nan, we need to ignore that time point or SVD will fail in tears. 
        plt.figure('svd')
        svd4 = svdfig.add_subplot(nplot-1,3,4)
        plt.ylabel('S0')
        a=do_svd_protected(qs[qroi],ts,S0[:,qroi],n=svd_n,smooth=smooth,logscan=logscan,fig='svd',sub=(nplot-1,3,4))
        
        plt.figure('svd')
        svd7=svdfig.add_subplot(nplot-1,3,7)
        plt.ylabel('S2')
        a=do_svd_protected(qs[qroi],ts,S2[:,qroi],n=svd_n,smooth=smooth,logscan=logscan,fig='svd',sub=(nplot-1,3,7)) 
    
    
    
    #### make figures look acceptable and save them ####
    plt.figure('res')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(figdir+basename+'_result.png')


    if show_svd:
        plt.figure('svd')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        svdfig.suptitle('SVD of %s, scanning %s, %i/%i events' %(basename,scanvar,numshots_used,numshots))
        plt.savefig('%s%s_SVD.png'%(figdir,basename))
    now = time.time() #Time after it finished
    print('done')
        
    
        
