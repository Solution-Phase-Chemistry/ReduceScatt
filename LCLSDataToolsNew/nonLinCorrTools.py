"""
Python example to correct for image spatial dependent non-linearity. 
Use function "getCorrectionFunc" in order to generate a correction 
function from a intensity dependence measurement of an identical 
signal.

T. B. van Driel, S. Herrmann, G. Carini, M. M. Nielsen, and H. T. Lemke, Correction of Complex Nonlinear Signal Response from a Pixel Array Detector, J Synchrotron Rad 22, 584 (2015).
DOI: 10.1107/S1600577515005536
"""

import numpy as np
import copy
from scipy import linalg

def iterfy(iterable):
    if isinstance(iterable, str):
        iterable = [iterable]
    try:
        iter(iterable)
    except TypeError:
        iterable = [iterable]
    return iterable

def polyVal(comps,i0):
    """Multidimensional version of numpy.polyval.  Evaluates polynomial."""                                                          
    i0 = np.asarray(iterfy(i0))                                             
    pol = np.vander(i0,len(comps))                                        
    return np.asarray(np.matrix(pol)*np.matrix(comps.reshape((len(comps),-1)))).reshape((len(i0),)+np.shape(comps)[1:])                            
                                                                                
def polyDer(comps,m=1):                                                         
    """Multidimensional version of numpy.polyder. Derivative of polynomial."""                                                          
    compsf = comps.reshape((len(comps),-1))
    n = len(compsf) - 1                                                            
    y = compsf.reshape((n+1,-1))[:-1] * np.expand_dims(np.arange(n, 0, -1),1)                      
    if m == 0:                                                                    
        val = comps
        return val
    else:                                                                         
        val = polyDer(y, m - 1) 
        return val.reshape((n,)+np.shape(comps)[1:])

def polyFit(i0,Imat,order=3, removeOrders=[]):
    """Multidimensional version of numpy.polyfit. Polynomial fit to data"""                                                          
    Imatf = Imat.reshape((len(Imat),-1))
    pol = np.vander(i0,order+1)                                                   
    removeOrders = iterfy(removeOrders)                                     
    removeOrders = np.sort(removeOrders)[-1::-1]                                  
    for remo in removeOrders:                                                     
        pol = np.delete(pol,-(remo+1),axis=1)                                       
    lhs = copy.copy(pol)                                                          
    scale = np.sqrt((lhs*lhs).sum(axis=0))                                        
    lhs /= scale                                                                  
    comps,resid,rnk,singv = linalg.lstsq(lhs,Imatf)                                
    comps = (comps.T/scale).T                                                     
                                                                                  
    for remo in removeOrders:                                                     
        comps = np.insert(comps,order-remo,0,axis=0)                                
    return comps.reshape((order+1,)+np.shape(Imat)[1:])


def getCorrectionFunc(dmat=None,i=None,ic=None,order=5,sc=None,search_dc_limits=None):
    """ 
    Create nonlinear correction function from a calibration dataset consiting of:
    dmat   	ND array of the reference patterns corresponding to values of i,
            The first dimension corresponds to the calibration intensity 
    values and has the same length as i.
    i     	array of intensity values (floats) of the calibration
    ic		A working point around which a polynomial correction will be
    developed for each pixel.
    order	the polynomial order up to which the correction will be 
            deveoped.
    sc		optional: calibration image at ic. default is the image at ic
    search_dc_limits	absolute limits around ic which are used to determine the 
        calibration value of ic as linear approximation of a short interval. 
    optional, can sometimes help to avoid strong deviations of the 
    polynomial approximatiuon from the real measured points.

    Returns corrFunc(D,i), a function that takes an ND array input for correction
            (1st dimension corresponds to the different intensity values) 
    as well as the intensity array i.
    """
        
    if search_dc_limits is not None:
        search_dc_limits = iterfy(search_dc_limits)
        if len(search_dc_limits)==1:
            msk = (i>i-np.abs(search_dc_limits)) & (i<i+np.abs(search_dc_limits))
        elif len(search_dc_limits)==2:
            msk = (i>i-np.min(search_dc_limits)) & (i<i+np.max(search_dc_limits))
        p0 = polyFit(i[msk],dmat[msk,...],2)
        dc = polyVal(p0,i0_wp)
        pc = polyFit(i-ic,Imat-dc,order,removeOrders=[0])
        pcprime = polyDer(pc)
        c = lambda i : polyVal(pc,i-ic) + dc
    else:
        pc = polyFit(i-ic,dmat,order,removeOrders=[])
        pcprime = polyDer(pc)
        c = lambda i : polyVal(pc,i-ic)
        dc = c(ic)
    c_prime = lambda i : polyVal(pcprime,i-ic)
    cprimeic = c_prime(ic)
    if sc is None:
        sc = c(ic)
    def corrFunc(D,i):
        i = np.asarray(i).ravel()
        return (sc.T/ic*i).T + cprimeic/c_prime(i)* sc/dc * (D-c(i))
    return corrFunc

# def corrFunctPhis(D, i):
#     corrected = np.zeros_like(D)
#     phi = D.shape[1]
#     for x in range(phi):
#         NewCorr = corrFunc(D[:,x,:], i)
#         corrected[:,x,:] = NewCorr
#     return corrected


# CorrArray = Azavs[:,:,:]
# params = []
# for x in range(len(phis)-1): 
#     dmat = yvals[:,x,:]
# #     print(dmat.shape)
#     nanfilt1 = np.isfinite(np.nanmean(dmat,0))
# #     print(nanfilt1.shape)
#     dmat1 = dmat[:, nanfilt1]
#     corrFunc, parm = getCorrectionFunc(dmat1, xVals, xVals[len(xVals)//2], order = 3, sc = None, search_dc_limits = None)
# #     new = corrFunc(dmat1, xVals)
#     params.append(parm)
#     even_newer = corrFunc(CorrArray[:,x,nanfilt1], Isum)
#     CorrArray[:,x,nanfilt1] = even_newer
#     print(x, 'done')
