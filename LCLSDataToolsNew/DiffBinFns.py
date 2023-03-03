### by Sumana Raj ###


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
from LCLSDataToolsNew.nonLinCorrTools import *

from sklearn.linear_model import RANSACRegressor as RSC
from sklearn.linear_model import LinearRegression as LR




def MakeScanAx(paramDict,outDict,tt_corrNew=None):
    ''' set up scan axis, use scanvariable or if time axis use time tool to calulate delays
    if tt_corrNew=None then use ttCorr from h5
    if ttcorrNew=(p1,p2) where p1,p2 are floats then calculate tt_corr with linear equation ttcorr = p1*fltpos + p2'''
    
    x_var=paramDict['x_var']
    scanvar=paramDict['scan_var']
    t0_corr=paramDict['t0_corr']
    use_tt=paramDict['use_TT']
    scanvec=outDict['h5Dict']['scan_vec']
    lxt=outDict['h5Dict']['lxt']
    enc=outDict['h5Dict']['encoder']
    if use_tt:
        ttCorr=outDict['h5Dict']['ttCorr']
    
    
    
    
    if x_var is None: # if no x specified, then use scan motor
        if scanvar=='newdelay': #or (scanvar=='lxt_ttc'):
            #we want time on the x-axis. two options:  ttool or no ttool:

            if t0_corr is None: #get additive correction for t0
                t0_corr=0.
            else:
                print('using provided t0_corr %f'%t0_corr)

            if (use_tt==False) or (use_tt=='filter'): # get delay info w/o time tool
                try:
                    x=t0_corr+lxt+enc*1e-12 # lxt plus encoder value is the time (seconds).
                    print('t = lxt+ (encoder)*1e-12')
                except KeyError:
                    try:
                        x=t0_corr+lxt
                        print('encoder unavailable, t = lxt')
                    except:
                        x=t0_corr+enc*1e-12
                        print('lxt unavailabe, t = (encoder)*1e-12')
                        
            elif use_tt=='withlxt':
                if tt_corrNew is not None:
                    ttCorr=tt_corrNew[0]*outDict['h5Dict']['ttFLTPOS']+tt_corrNew[1]
                try:
                    x=t0_corr+lxt+(enc+ttCorr)*1e-12
                    print('t=lxt+(encoder + tt/ttCorr)*1e-12')
                except:
                    x=t0_corr+(enc+ttCorr)*1e-12
                    print('t=(encoder + tt/ttCorr)*1e-12')
            
            else: #want to use the timetool
                if tt_corrNew is not None:
                    ttCorr=tt_corrNew[0]*outDict['h5Dict']['ttFLTPOS']+tt_corrNew[1]
                    
                x=t0_corr+(enc+ttCorr)*1e-12
                print('t=(encoder + tt/ttCorr)*1e-12')
                    
                    
        else: #any other scanvar besides time: just take the values from the array
            x=scanvec
            print('using scan_vector for binning axis')

    else: # want to bin along a non-scanned variable
        x=d[x_var][:]
        print('using x_var for binning axis')
        
    outDict['xs']=x
    

    
    
def DarkSubtract(paramDict,outDict):
    
    f_xoff=outDict['filters']['f_xoff']
    azav_temp=outDict['h5Dict']['azav']

    #### subtract mean of dark curves ######    
    dark = azav_temp[f_xoff, :,:]
    darkMean = np.nanmean(dark,0)
    azav_temp = azav_temp - darkMean

    azav_temp=outDict['h5Dict']['azav']
    
    print('x-ray off subtraction done!')
    
    

    
    
def NormalFactor(paramDict,outDict):
    ''' make normalization factor and save normalized cake'''
    
    print('normalize data')
    
    qnorm=paramDict['qnorm']
    Isum=outDict['Iscat']
    cspad_azav=outDict['h5Dict']['azav']
    f_intens=outDict['filters']['f_good']
    f_loff=outDict['filters']['f_loff']
    qs=outDict['h5Dict']['qs']
    
    #### how to normalize: by the Isum, or by a section of the high Q range?
    if qnorm is None:
        normal_factor=Isum
        normal_factor_e=normal_factor

    else:
        qlow=qnorm[0]
        qhigh=qnorm[1]
        # normal_factor=highq_normalization_factor(cspad_azav, qs, qlow,qhigh)
        normal_factor=norm_factor_phis(cspad_azav, qs, qlow,qhigh)
        
        
        
    ####### save 300 off shots averaged as cake#####
    early_x=np.where(f_intens&f_loff)[0][:300]
    assert len(early_x)>1, "There are no valid laser-off shots; lightStatus/laser is %s"%str(d['lightStatus']['laser'][:20])
#             print(f_lon)
#             print(f_loff)
    #early_x=(np.argsort(x)<300)&(f_intens&f_loff)
    # cake=np.nanmean(cspad_azav[early_x,:,:]/normal_factor[early_x,None,None],0) #normalize by norm 
    cake=np.nanmean(divAny(cspad_azav[early_x,:,:],normal_factor[early_x,:]),0) #normalize by norm
    cake=divAny(cake,np.nanmax(cake,1))
    
    outDict['normal_factor']=normal_factor
    outDict['loff_cake']=cake
    
    print('normalize data done!')
    
    ### apply normal_factor ###
    
    cspad_azav=divAny(cspad_azav,normal_factor)
    outDict['h5Dict']['azav']=cspad_azav
    print('normalization applied')
    
    if paramDict['useAzav_std']=='WAve':
        outDict['h5Dict']['azav_std']=divAny(outDict['h5Dict']['azav_std'],normal_factor)    
    
    
    
    
    
    
    
    
    
def EnergyCorr(paramDict,outDict):
    '''apply photon energy correction using SVD, use after normalization.'''
    
    print('applying energy correction')
    ebeam=outDict['h5Dict']['ebeam_hv']
    f_xon=outDict['filters']['f_xon']
    f_intens=outDict['filters']['f_good']
    f_loff=outDict['filters']['f_loff']
    azav_temp=outDict['h5Dict']['azav']
    Isum=outDict['Iscat']
    qs=outDict['h5Dict']['qs']

    energy_corr_2d=SVDcorrection(azav_temp,ebeam,qs,filt=(f_xon&f_intens&f_loff),n=1,binp=100,poly=2)
 
    cspad_azav=azav_temp-energy_corr_2d
    
    outDict['h5Dict']['azav']=cspad_azav

    


    

    
    
    
    
    
def DetectorNonlinCorr(paramDict,outDict):
    ''' use after normalization '''
    
    NonLinCorr=paramDict['NonLin_corr']
    cspad_azav=outDict['h5Dict']['azav']
    qs=outDict['h5Dict']['qs']
    phis=outDict['h5Dict']['phis']
    Isum=outDict['Iscat']
    f_xon=outDict['filters']['f_xon']
    f_intens=outDict['filters']['f_good']
    f_loff=outDict['filters']['f_loff']
    
 ####### detector nonLinCorr code ########
    if NonLinCorr=='poly': 
        print('do polynomial nonlinear corrections')
        ##### Polynomial NonLinCorrection
        off_xray=np.where(f_xon&f_intens&f_loff)[0][:]
        offshot = cspad_azav[off_xray, :, :]

        xVals, yvals = BinnedMeanCake(Isum[off_xray],offshot,100)
        CorrArray = np.full_like(cspad_azav, np.nan)
#             params = []
        for phibin in range(len(phis)-1): 
            dmat = yvals[:,phibin,:]
            nanfilt1 = np.isfinite(np.nanmean(dmat,0))
            dmat1 = dmat[:, nanfilt1]
            corrFunc = getCorrectionFunc(dmat1, xVals, xVals[len(xVals)//2], order = 3, sc = None,
                                         search_dc_limits = None)
        #     new = corrFunc(dmat1, xVals)
        #     params.append(parm)
            even_newer = corrFunc(cspad_azav[:,phibin,nanfilt1], Isum)
#                 print(even_newer.shape, CorrArray.shape, CorrArray[:, phibin, nanfilt1].shape)
            CorrArray[:,phibin,nanfilt1] = even_newer
            print(phibin, 'done')
        cspad_azav=CorrArray


    elif NonLinCorr=='SVD':
#             if NonLinCorr:
        print('do SVD nonlinear corrections')
        #### SVD Nonlinear correction   
        
        nonLin_corr_2d=SVDcorrection(cspad_azav,Isum,qs,filt=(f_xon&f_intens&f_loff),n=1,binp=100,poly=2)
  
        cspad_azav=cspad_azav-nonLin_corr_2d

    elif NonLinCorr == 'SVDbyBin':
        print('do SVD nonlinear corrections for each phi bin')
        
        nonLin_corr_3d=SVDcorrectionByBin(cspad_azav,Isum,qs,
                                          filt=(f_xon&f_intens&f_loff),n=1,binp=100,poly=2)
   

        cspad_azav=cspad_azav-nonLin_corr_3d

#                 print('done')

    outDict['h5Dict']['azav']=cspad_azav
    print('nonlinear correction - done!')
    
    
    
    
    
    

        
    
    
    
    
    
    
    
def doDifference(paramDict,outDict):
    print('starting difference signal')
    
    adj_subtr=paramDict['AdjSub']
    cake=outDict['loff_cake']
    cspad_azav=outDict['h5Dict']['azav']
    normal_factor=outDict['normal_factor']
    f_intens=outDict['filters']['f_good']
    f_loff=outDict['filters']['f_loff']
    f_lon=outDict['filters']['f_lon']
    qerr=paramDict['useAzav_std']
    x=outDict['xs']
    
    ### do the diff signal ###
    totaloff=np.nanmax(np.nanmean(cake,0))
#                 totaloff = 1
    diff=DifferenceSignal(cspad_azav,f_intens,
                          f_lon,f_loff,adj_subtr,totaloff=totaloff) 
    
    yerr=None
    if qerr=='WAve':
        diff_err=DifferenceError(cspad_azav,outDict['h5Dict']['azav_std'],f_intens,
                               f_lon,f_loff,adj_subtr,totaloff=totaloff)
        onErr=diff_err[f_intens&f_lon]
        outDict['diff_Err']=onErr

    xData=x[f_intens&f_lon]
    onData=diff[f_intens&f_lon]
    
    outDict['x_Data']=x[f_intens&f_lon]
    outDict['diff_Data']=diff[f_intens&f_lon]
    
    
    print('Difference Signal Done!')
    
    
    
    

    
def AveAllShots(paramDict, outDict):
    ''' Averages difference signal for all shots, returns a 1xqxphi array
    use this for example, if you have only one time delay at long times'''
    
    print('Averaging all shots difference signals')
    
    #inputs
    qerr=paramDict['useAzav_std']
    onData=outDict['diff_Data']
    
    if qerr=='WAve':
        yerr=outDict['h5Dict']['azav_std']
        diff_bin, diff_std= WAve(onData,yerr,axis=0)
        
    else:
        diff_bin=np.nanmean(onData,axis=0)
        diff_std=StandErr(onData,0)
    
    diff_bin=np.expand_dims(diff_bin,axis=0)
    diff_std=np.expand_dims(diff_std,axis=0)
    outDict['diff_bin']=diff_bin
    outDict['diff_std']=diff_std
    outDict['xcenter']=[0]
    outDict['xbin_occupancy']=onData.shape[0]
    
    print('Average done!')
    
    

    
    
    
def doTimeBinning(paramDict,outDict):
    # 'binMethod' : 'ave', ## 'ave' or 'sum' ? 
    
    print('Starting binning')
    
    #inputs
    xstat=paramDict['xstat']
    qerr=paramDict['useAzav_std']
    
    xData=outDict['x_Data']
    onData=outDict['diff_Data']
    
    #qerr?
    if qerr=='WAve':
        qerr=True
        yerr=outDict['h5Dict']['azav_std']
    else:
        qerr=False
        yerr=None
    pts_per_bin=0
    
    #do binning
    if paramDict['binSetup']=='points':
        pts_per_bin=paramDict['binSet2']
        print('bin by points, %i per bin'%pts_per_bin)
        bin_outfile=BinStat(xData,onData,yerr=yerr,n=pts_per_bin,binByPoints=True,
                 showplot=False,set_bins=None,count=True, xstat=xstat)
        
    elif paramDict['binSetup']=='nbins':
        nbins=paramDict['binSet2']
        print('bin by number of bins, %i bins'%nbins)
        bin_outfile=BinStat(xData,onData,yerr=yerr,n=nbins,binByPoints=False,
                 showplot=False,set_bins=None,count=True, xstat=xstat)
        
    elif paramDict['binSetup']=='fixed':
        centers=paramDict['binSet2']
        print('bin by given bins')#: %s'%str(set_bins))
        bin_outfile=BinStat(xData,onData,yerr=yerr,n=pts_per_bin,binByPoints=True,
                                     showplot=False,set_bins=centers,count=True, xstat=xstat)
        
    elif paramDict['binSetup']=='unique':
        print('bin by unique x axis value')
        bin_outfile=BinStatbyAxis(xData,onData,yerr=yerr,n=pts_per_bin,binByPoints=True,
                                  showplot=True,set_bins=None,count=True)
    
    #binning outputs
    print(bin_outfile.keys())
    try:
        outDict['xbin_occupancy']=bin_outfile['bincount']
    except:
        print('bincount not saved')
    diff_temp=bin_outfile['binmean']
    ds_temp=bin_outfile['binstd']
    outDict['xcenter']=bin_outfile['xcenter']
    if xstat:
        outDict['xmean']=bin_outfile['xmean']
        outDict['xstd']=bin_outfile['xstd']
    
    if len(diff_temp.shape)==2:
        diff_temp=np.expand_dims(diff_temp,axis=1)
        ds_temp=np.expand_dims(ds_temp,axis=1)
    outDict['diff_bin']=diff_temp
    outDict['diff_std']=ds_temp
    
    if qerr:
        qerr_temp=bin_outfile['qerr']
        if len(diff_temp.shape)==2:
            qerr_temp=np.expand_dims(qerr_temp,axis=1)
        outDict['diff_azav_std']=qerr_temp
    
    print('binning - done!')
    
    
    
    

    
    
    
    

def SVDcorrection(data,var1,qs,filt=None,n=1,binp=100,poly=2):
    '''correct data (dimensions t x phi x q) using svd( data - median(data) ) binned wrt variable var1 (dimension t),  correction of n=1 means only first component of svd is used to correct
    returns correction to be subtracted.
    data input is filtered by filt; data should already be normalized!!
    binp=points per bin'''
    
    if filt is None:
        filt=np.full_like(var1,1,dtype=bool)
    
    xData=var1[filt&~np.isnan(var1)]
    onData=np.nanmean(data,1) #average over phis
    onData=onData[filt&~np.isnan(var1)]
    
    #bin data wrt var1
    outfile=BinStat(xData,onData,binp,yerr=None,binByPoints=True,showplot=False,set_bins=None,count=True, xstat=True)
    xBin=outfile['xcenter']
    dBin=outfile['binmean']
    
    ss,ww,tt=do_svd_protected(qs[10:-10],xBin,(dBin-np.nanmedian(dBin,0))[:,10:-10])
    plt.suptitle('SVD of data - median')
    
    corr2D=np.zeros((data.shape[0],1,data.shape[2]))
    
    for nn in range(n):
        corrSS=np.full(qs.shape,np.nan)
        corrSS[10:-10]=ss.T[:,nn]*ww[nn]

        corrParams=np.polyfit(xBin[1:-2],tt[1:-2,nn],poly)
        corrF=np.polyval(corrParams,var1)
        corr2D+=(corrSS*corrF[:,None])[:,None,:]
        
    
    return corr2D




def SVDcorrectionByBin(data,var1,qs,filt=None,n=1,binp=100,poly=2):
    '''correct data (dimensions t x phi x q) using svd( data - median(data) ) binned wrt variable var1 (dimension t),  correction of n=1 means only first component of svd is used to correct
    returns correction to be subtracted. Does this for each phi bin.
    data input is filtered by filt; data should already be normalized!!
    binp=points per bin'''
    
    if filt is None:
        filt=np.full_like(var1,1,dtype=bool)
    
    xData=var1[filt&~np.isnan(var1)]
    onData=data[filt&~np.isnan(var1)]
    
    #bin data wrt var1
    outfile=BinStat(xData,onData,binp,yerr=None,binByPoints=True,showplot=False,set_bins=None,count=True, xstat=True)
    xBin=outfile['xcenter']
    dBin=outfile['binmean']

    corr3D=np.zeros((data.shape))
    for pp in range(dBin.shape[1]):
        dBinPhi=dBin[:,pp,:].squeeze()
        nanfilt1 = np.isfinite(np.nanmean(dBinPhi,0)) #filter out qs that are not finite
        ss,ww,tt=do_svd_protected(qs[nanfilt1],xBin,(dBinPhi[:,nanfilt1]-np.nanmedian(dBinPhi[:,nanfilt1],0)))
        plt.suptitle('SVD of data - median')
    
        corr2D=np.zeros((data.shape[0],1,data.shape[2]))
        
        for nn in range(n):
            corrSS=np.full(qs.shape,np.nan)
            corrSS[nanfilt1]=ss.T[:,nn]*ww[nn]

            corrParams=np.polyfit(xBin[1:-2],tt[1:-2,nn],poly)
            corrF=np.polyval(corrParams,var1)
            corr2D+=(corrSS*corrF[:,None])[:,None,:]
        corr3D[:,pp,:]=corr2D.squeeze()
    
    return corr3D    




 