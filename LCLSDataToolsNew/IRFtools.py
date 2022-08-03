'''based on multiparameter maximum likelihood analysis framework as in DOI:10.1002/anie.200900741'''


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import h5py
import sys
import os
import scipy.io as scio

from scipy import interpolate as sintp
from scipy.signal import deconvolve,convolve
from scipy.optimize import curve_fit
import scipy.ndimage as scnd

from numpy.polynomial import polynomial as P
from multiprocessing import Pool

from LCLSDataToolsNew.anisotropyToolsAll import *
from LCLSDataToolsNew.plottingTools import *



def IRFprep(ddata,qs=None,ts=None,ref=None,tref=None,qCenter=1.5,qWidth=.15,
           trange=None,qrange=None):
    ''' loads 2d ddata (ts x qs), selects region from qCenter-qWidth to qCenter+qWidth,
    calculate mean and standard deviation and plot along with reference.
    plot ranges given by trang and qrange
    '''
    
    ########plot t traces for given q and qwidth
    qq=qCenter
    qWidth=qWidth
    S2=ddata

    if trange is not None: 
        goodt= np.where((ts > trange[0]) & (ts < trange[1]))[0]
    else:
        goodt=np.full_like(ts,True)
    if qrange is not None:
        goodq= np.where((qs > qrange[0]) & (qs < qrange[1]))[0]
    else:
        goodq=np.full_like(qs,True)

    ###### 2D plot with dashed lines for qq and qWidth
    fig,ax=plt.subplots(nrows=1,ncols=2,num='q slices',figsize=(15,6))
    plot_2d(ts[goodt],qs[goodq],S2[goodt,:][:,goodq].squeeze(),fig='q slices',sub=(1,2,1))
    ax[0].set_title('$ \Delta S_2 $')
    ax[0].axhline(qq,color='k')
    ax[0].axhline(qq-qWidth,ls='--',color='k')
    ax[0].axhline(qq+qWidth,ls='--',color='k')

    ##### make t trace for qq +/ qWidth 
    qRange2=(qq-qWidth,qq+qWidth)
    qTemp=np.nonzero((qs>qRange2[0])& (qs<qRange2[1]))
    S2_slice=np.nanmean(S2[:,qTemp].squeeze(),1)
    S2s=np.abs(S2_slice)
    S2_std=np.nanstd(S2[:,qTemp].squeeze(),1)

    ### plot t trace
    plt.subplot(1,2,2)
    plt.errorbar(ts[goodt],S2s[goodt]/np.nanmax(S2s),yerr=S2_std[goodt]/np.nanmax(S2s),color='k',label='q= %.2f $A^{-1}$'%qq)
    if ref is not None:
        plt.plot(tref,ref,'r',label='Reference')
    plt.xlabel('time (ps)')
    # plt.ylabel('$| \Delta S_2 |$')
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.legend()
    ax[1].set_title('each trace is the average for q +/ %.2f $A^{-1}$'%(qWidth))

    return S2s, S2_std



def chi2(CalcSignal,ExpSignal,sigma,N,p):
    '''N = number of q points, p= number of free parameters, 
    sigma= standard deviationo at each q point 
    returns chi2'''
    temp1=np.full((N),np.nan)
    for i in range(N):
        temp1[i]=(CalcSignal[i]-ExpSignal[i])**2/2/sigma[i]**2
    cchi2=np.nansum(temp1)/(N-p-1)
    return cchi2

def likelihood(chi2):
    return np.exp(-1*chi2)

def fitIRFtt0(S2N,ErrN,Ref2,tt0L,sigma,tnew1,goodt):
    ''' for each tt0 option calculate chiAll'''
    chiAll=np.full(tt0L.shape,np.nan)
    for ii,tt0 in enumerate(tt0L):
        testIRF=gaus(tnew1,1,tt0,sigma,0)
        aa=convolve(Ref2,testIRF,mode='same')
        aaN=aa/np.nanmax(aa)
        chiAll[ii]=chi2(aaN[goodt],S2N[goodt],ErrN,goodt.shape[0],2)
    return chiAll


def gaus(x, a, b, c, d):
    return a * np.exp(-1*(x-b)**2/2/c**2) + d


def doIRFfit(S2s,S2_std,qs=None,ts=None,ref=None,tref=None,tt0L=None,sigmaL=None,
           trange=None):
    '''calculate IRF and chi2 for all t0,sigma combinations from t0L and sigL
    '''
    


    #### interpolate all data to have matching t bins given by tnew1 ####

    if np.nanmax(ts)>0.9:
        tn_max=0.9
    else:
        tn_max=np.nanmax(ts)
    tnew1=np.linspace(np.nanmin(ts),tn_max,200)
    # tnew1=np.linspace(-.3,1,200)

    RefIntp=sintp.interp1d(tref,ref)
    S2Intp=sintp.interp1d(ts,S2s)
    stdIntp=sintp.interp1d(ts,S2_std)

    Ref2=np.full_like(tnew1,0)
    Ref2[tnew1>0]=RefIntp(tnew1[tnew1>0])
    S2new=S2Intp(tnew1)
    S2N=S2new/np.nanmax(S2new)
    ErrN=stdIntp(tnew1)/np.nanmax(S2new)



    ###### do the calculation #######
    chiAll=np.full((sigmaL.shape[0],tt0L.shape[0]),np.nan)
    print('chiAll shape',chiAll.shape)

    # trange=(-.4,0.9)
    goodt= np.where((tnew1 > trange[0]) & (tnew1 < trange[1]))[0]
    
    print('starting calculation')
    #### do the calculation in parallel ####
    pool = Pool(processes=12)
    results = []
    chiAll=np.full((sigmaL.shape[0],tt0L.shape[0]),np.nan)
    poolScan=range(sigmaL.shape[0])
    for jj in poolScan:
        results.append(pool.apply_async(fitIRFtt0, args=(S2N,ErrN,Ref2,tt0L,
                                                         sigmaL[jj],tnew1,goodt)))
    pool.close()
    pool.join()

    for jj in poolScan:
        outTemp= results[jj].get()
        chiAll[jj,:] = outTemp

    print('finished calculation')
    
    ### plot results ###
    fig,ax=plt.subplots(nrows=1,ncols=3,num='chiAll',figsize=(18,6))
    plot_2d(sigmaL,tt0L,chiAll,fig='chiAll',sub=(1,3,1))
    plt.xlabel('sigma (ps)')
    plt.ylabel('t0 (ps)')
    ax[0].set_title('Chi $^{2}$')

    args=np.nonzero(chiAll==np.nanmin(chiAll))
    print('chi min %.4f' %np.nanmin(chiAll),
          'sigma',sigmaL[args[0]],'fwhm',sigmaL[args[0]]*2.355,
          'tt0',tt0L[args[1]])

    outIRF=gaus(tnew1,1,tt0L[args[1]],sigmaL[args[0]],0)
    outaa=convolve(Ref2,outIRF,mode='same')
    outaaN=outaa/np.nanmax(outaa)

    plt.subplot(1,3,2)
    plt.plot(tnew1,S2N,label='S2')
    plt.plot(tnew1,outaaN,'b--',label='convolution')
    plt.plot(tnew1,Ref2,label='Reference')
    plt.plot(tnew1,outIRF,label='IRF')
    plt.xlim(-1,1)
    plt.xlabel('t (ps)')
    ax[1].set_title('Best Fit')
    plt.legend()

    ######### calculate and plot likelihood ####################
    LLall=likelihood(chiAll)

    t0range2=(tt0L[0],tt0L[-1])
    ssrange2=(sigmaL[0],sigmaL[-1])
    t0good2=np.nonzero((tt0L>t0range2[0]) &(tt0L<t0range2[1]))
    ssgood2=np.nonzero((sigmaL>ssrange2[0]) &(sigmaL<ssrange2[1]))

    LLall2=LLall[:,t0good2][ssgood2,:].squeeze()

    plot_2d(sigmaL[ssgood2],tt0L[t0good2],LLall2,fig='chiAll',sub=(1,3,3))
    plt.xlabel('sigma (ps)')
    plt.ylabel('t0 (ps)')
    ax[2].set_title('Likelihood')
        
        
    return chiAll, LLall2

