import numpy as np
from scipy.interpolate import interp1d
from multiprocessing import Pool

from matplotlib import pyplot as plt
import matplotlib as mpl
import h5py
import sys
import os
import time
import scipy
import scipy.io as scio
from scipy.optimize import curve_fit

def recalcQ_all(outDict):
    ebeam=outDict['h5Dict']['ebeam_hv']
    qs=outDict['h5Dict']['qs']
    azav=outDict['h5Dict']['azav']
    setE=outDict['h5Dict']['set_ebeam_keV']
    d2sam=outDict['h5Dict']['set_d2sam_cm']
    
    pool = Pool(processes=20)
    poolScan=range(azav.shape[0])
    results=[]
    
    for ii in poolScan:
        results.append(pool.apply_async(recalcQforEbeam,args=(qs,azav[ii,:,:],ebeam[ii],setE,d2sam)))
    pool.close()
    pool.join()

    azavNew=np.full_like(azav,np.nan)
    for ii in poolScan:
        azavNew[ii,:,:]=results[ii].get()
        
    outDict['h5Dict']['azav']=azavNew

def recalcQforEbeam(qs,azav,ebeam,setE,d2sam):
    ''' for a single shot, recalculate q axis based on ebeam, interpolate azav based on corrected axis;
        azav size phis x qs
    '''
    setLam=1239.8/(setE*1000)*10 #in angstroms
    Lam=1239.8/(ebeam)*10
    
    theta1=np.arcsin(qs*setLam/4/np.pi)
    RR=d2sam*np.tan(2*theta1)

    theta2=np.arctan(RR/d2sam)/2
    qsNew=np.sin(theta2) *4*np.pi/Lam

    azavNew=np.full_like(azav,np.nan)
    for ii in range(azav.shape[0]):
        tempI=interp1d(qsNew,azav[ii,:],bounds_error=False, fill_value=np.nan)
        azavNew[ii,:]=tempI(qs)
        
    return azavNew