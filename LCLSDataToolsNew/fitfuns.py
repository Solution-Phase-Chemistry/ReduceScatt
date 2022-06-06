import scipy
import matplotlib
from matplotlib import pyplot as plt
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
import math

def fit_plot(func,x,y,p0=None,labels=None):
    plt.figure()
    plt.plot(x, func(x,*p0),'b-', label="Guess")
    plt.plot(x, y, 'ko', label="Data")
    try:
        popt, pcov = curve_fit(func, x, y,p0)
        plt.plot(x, func(x, *popt), 'r-', label="Fit")
        fitted=func(x,*popt)
        plt.plot(x, y-fitted,'g-', label="Residual")
    except RuntimeError:
        print('Maximum attempts exceeded')
        popt=[]
        fitted=[]
        pcov=[]
    plt.legend()
    print_params(popt,labels)
    print('R_squared: ' + str(R_squared(func,x,y,popt)))
    return popt,pcov,fitted

def inst_response(timeaxis, fwhm, showplot=False):
    timezero = np.argmin(abs(timeaxis)) #find the index of the closest point to time zero.
    #print(timezero)
    timestep=np.diff(timeaxis)[0] #x-axis time step, seconds
    if ~(np.sum(~np.isclose(np.diff(timeaxis),timestep))==0):
        raise ValueError('improperly interpolated data; must have uniform x axis.')
    sigma = fwhm/(2*np.sqrt(2*np.log(2))) #standard deviation in units of seconds
    #print(shift_abs)
    time_x = (timestep*np.arange(-timezero,timezero))
    unit_gaussian = np.exp(-time_x**2/(2*sigma**2))
    #unit_gaussian = sig.gaussian(pts,sigma_steps)
    integ=np.sum(unit_gaussian)
    unit_gaussian = unit_gaussian / integ
    if showplot:
        plt.figure()
        plt.plot(time_x, unit_gaussian)
    return unit_gaussian

def sequential_kinetic(xx,t0,s,t1,t2,a1,a2,return_pops=False):
    #We assume that the time constant for populating the first excited state is faster than the IRF
    '''
#     offset - constant background
#     scale - proportionality constant (global)
    t0 - shift constant (t0)
    s - FWHM of the broadening gaussian
    t1 - 1->2 time constant 1
    t2 - 2->3 time constant 2
    a1 - amplitude 1
    a1 - amplitude 2
    '''
    
    #define half-range for convolution function
    c = s/(2*np.sqrt(2*np.log(2)))
    #define rate constants
    k1 = 1/t1
    k2 = 1/t2
    
    pop1 = 0.5*np.exp(k1*(t0-xx)+(k1*c)**2/2.0)*(1+erf((xx-(t0+k1*c**2))/(np.sqrt(2)*c)))
    pop2 = 0.5*k1/(k2-k1)*(np.exp(k1*(t0-xx)+(k1*c)**2/2.0)*(1+erf((xx-(t0+k1*c**2))/(np.sqrt(2)*c)))-\
        np.exp(k2*(t0-xx)+(k2*c)**2/2.0)*(1+erf((xx-(t0+k2*c**2))/(np.sqrt(2)*c))))
    pop_gs = 1.0-pop1-pop2
    
    # construct the time dependent difference trace
    dif_out = -(a1*pop1+a2*pop2)
    pop_out = [pop1,pop2,pop_gs]

    if return_pops:
        return pop_out,dif_out
    else:
        return dif_out
    
def sequential_kinetic_numeric(xx,t0,s,t1,t2,a1,a2,return_pops=False):
    labels=['t0','s','t1','t2','a1','a2']
    #numerically convolute instead of algebraically, so this can be used for other models
    #We assume that the time constant for populating the first excited state is faster than the IRF
    '''
#     offset - constant background
#     scale - proportionality constant (global)
    t0 - shift constant (t0)
    s - FWHM of the broadening gaussian
    t1 - 1->2 time constant 1
    t2 - 2->3 time constant 2
    a1 - amplitude 1
    a2 - amplitude 2
    '''
    
    #define convolution function
    resp=inst_response(xx,s,False)
    
    #define longer x-axis to get the end to not be weird due to numerics
    stepsize=np.diff(xx)[0]
    x=np.concatenate((xx,np.amax(xx)+np.arange(1,100)*stepsize))
    
    #define step function for initial excitation
    step=np.zeros(np.shape(x))
    step[(x-t0)>0]=1
    
    #define rate constants
    k1 = 1/t1
    k2 = 1/t2
    
    pop1 = step*np.exp(-k1*(x-t0))
    pop2 = step*k1/(k2-k1)*(np.exp(-k1*(x-t0))-np.exp(-k2*(x-t0)))
    
    
    #remove extra ends created by convolution (1/2 the length of the irf on each side)
    begin=np.size(resp)/2
    end=np.size(xx)+begin
    
    pop1=sig.convolve(pop1,resp)[begin:end]
    pop2=sig.convolve(pop2,resp)[begin:end]
    pop_gs = 1.0-pop1-pop2
    
    # construct the time dependent difference trace
    dif_out = -(a1*pop1+a2*pop2)
    pop_out = [pop1,pop2,pop_gs]

    if return_pops:
        return pop_out,dif_out
    else:
        return dif_out

def inst_response_asym(timeaxis, fwhm, m, showplot=False):
    #m in seconds
    timeaxis = timeaxis-m
    timezero = np.argmin(abs(timeaxis)) #find the index of the closest point to time zero.
    shift_abs=np.amin(abs(timeaxis)) #magnitude of closest point to time zero (sub-point precision)
    #print(timezero)
    timestep=np.diff(timeaxis)[0] #x-axis time step, seconds
    sigma = fwhm/(2*np.sqrt(2*np.log(2))) #standard deviation in units of seconds
    #print(shift_abs)
    time_x = (timestep*np.arange(-timezero,timezero))+shift_abs
    unit_gaussian = np.exp(-time_x**2/(2*sigma**2))
    #unit_gaussian = sig.gaussian(pts,sigma_steps)
    integ=np.sum(unit_gaussian)
    unit_gaussian = unit_gaussian / integ
    if showplot:
        plt.figure()
        plt.plot(time_x, unit_gaussian)
    return unit_gaussian

def conv_exp_decay(x,s,t,m,scale):
    resp=inst_response_asym(x,s,m,False) #shift over instrument response by timezero # of points; this is movable by parameter m
    b=np.exp(-x/t)
    bc=sig.convolve(b,resp)[0:np.size(b)]
    f=-bc
    return (scale/2)*(f)

def conv_exp_grow(x,s,t,m,scale):
    resp=inst_response_asym(x,s,m,False)
    a=-np.ones(np.size(x))#step(x)
    b=1-np.exp(-x/t)
    ac=sig.convolve(a,resp)[0:np.size(a)]
    bc=sig.convolve(b,resp)[0:np.size(a)]
    f=ac+bc
    return (scale/2)*(f)

#def conv_double_exp(x,s,t1,t2,m,scale1,scale2):
#    ac=conv_exp_decay(x,s,t1,m,scale1)
#    bc=conv_exp_grow(x,s,t2,m,scale2)
#    return(ac+bc)

def print_params(popt,labels):
    if labels==None:
        return None
    else:
        for i, label in enumerate(labels):
            print(label + ': '+str(popt[i]))
            
def R_squared(funct, xdata, ydata, popt):
    residual=np.sum((ydata-funct(xdata,*popt))**2)
    total=np.sum((ydata-np.mean(ydata))**2)
    return 1.0 - (residual/total)

def expfit(x,a,b,c,d):
    labels=['a1','tau1','a2','tau2']
    return a*np.exp(-x/b) + c*np.exp(-x/d)