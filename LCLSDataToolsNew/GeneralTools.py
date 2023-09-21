import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic
from scipy import interpolate as inp
import warnings


def chooseR(low,high,dataA):
    ''' for 1D array dataA, returns indices for values of dataA within limits low, high.  Also returns low and high as a list '''
    Inds=np.nonzero((low < dataA) & (dataA < high) ) [0] 
    return Inds, (low,high)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx 


def divAny(lg,sml,axis=None):
    ''' divide any N dimension array by another array with 1 < dimensions < N-1.
    axis =  order of axes in lg array for division.   for broadcasting arrays, final dimensions should be equal.  
    Order of final dimensions should match order of sml dimensions! (eg lg=txqxphi, sml=phixq then axis=(0,2,1))
    if axis = None, finds equal axes and does division.  Will return an error if dimensions are not unique. 
    Returns array with dimensions of lg''' 
    tL=lg
    tS=sml
    sL=np.array(tL.shape)
    sS=np.array(tS.shape)
    
    if axis is None:
        #are dimensions unique?
        if sL.size != np.unique(sL).size or sS.size != np.unique(sS).size:
            print('dimensions are not unique, specify axes for division')
            return
        else:
            #find common dimensions and their indices
            val,inL,inS=np.intersect1d(sL,sS,assume_unique=True, return_indices=True)
            if inS.size != sS.size:
                print('dimensions of small do not match dimensions of large!')
                return
            else:
                iLall=np.arange(sL.size)
                iLdiff=np.delete(iLall,inL) #indices of sL for dimensions not found in sml
                iLb=np.concatenate((iLdiff,inL)) # reorder axes for broadcasting arrays, final dimensions should be equal
                tLb=np.transpose(tL,iLb) #reorder arrays
                tSb=np.transpose(tS,inS)
                
                
    else:
        iLb=np.array(axis)
        tLb=np.transpose(tL,iLb)
        tSb=tS 
        
    dd=tLb/tSb #divde
    iD=np.argsort(iLb) #find indices for returning to original order
    out=np.transpose(dd,iD) #back go original order

    return out
   

                


    
    
    
############ fun with errors #################


def ErrPropS(B,axis=None):
    ''' for Nd array a1,a2,a3 that that are added or subtracted, propagate the standard deviations given in B 
    where b1 is error for a1. Calculates along axis=(0,1)etc. '''
    C=np.sqrt(np.nansum(B**2,axis=axis))
    return C

def ErrPropM(A,B,qq,axis=None):
    ''' for Nd array A that contains the values being multiplied/divided,
    propagate the standard deviations given in B, where b1 is error for a1.
    qq is the product/quotient. Calculates along axis=(0,1)etc.'''
    C=qq*np.sqrt(np.nansum((B/A)**2,axis=axis))
    return C

def WAve(A,B,axis=None):
    ''' for Nd array A that is to be averaged, calculates weighted average based on errors B.
    Returns weighted average C and uncertainty Cerr.Calculates along axis=(0,1)etc.'''
    W=1/B**2 #weights
    W[np.nonzero(np.isnan(A))]=np.nan #want weights for nan to also be nan 
    temp=np.nansum(W,axis=axis)
    C=np.nansum(W*A,axis=axis)
    if type(C)==np.ndarray:
        C[np.nonzero(np.isnan(np.nanmean(W*A,axis=axis)))]=np.nan #nansum of nan returns 0 so fix that
    C=C/temp
    
    Cerr=1/np.sqrt(temp)
    
    return C,Cerr

    
def StandErr(A,axis=None):
    ''' for array A calculate standard error. Calculates along axis=(0,1)etc. if Axis=None, calculate over all dimensions.'''
    size=0
    if axis == None:
         B=np.nanstd(A)/np.sqrt(A.size)
    else:
        if type(axis) is not int:
            for i in axis:
                size+=A.shape[i]
        else:
            size=A.shape[axis]
        B=np.nanstd(A,axis=axis)/np.sqrt(size)
    return B


  
######################  other tools #####################  
def ShiftCalc(bins,shift_ang):
    ''' return shift_n for given number of phi bins and given shift_angle '''
    diff=np.zeros((bins,1))

    phi_space=360/bins

    for n in range(bins):
        diff[n]=abs(shift_ang-n*phi_space)

    shift_n=np.nonzero(diff==np.min(diff))[0][0]

    return shift_n


def TwoTheta2q(tthet,lam):
    '''for two theta in degrees and lambda=x-ray wavelength in angstroms, returns q in A-1'''
    q=np.sin(tthet/2*np.pi/180)*4*np.pi/lam

    return q

  
def q2twoTheta(q,lam):
    '''for q in A-1 and lambda=x-ray wavelength in angstroms, return 2theta in degrees'''

    tthet=np.arcsin(q*lam/4/np.pi)*2*180/np.pi

    return tthet

  
  
  
  
