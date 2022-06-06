import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import medfilt


def do_svd(qs,ts,data,n=5,smooth=None,showplot=True, fig=None, sub=None,logscan=False):
    '''Do an SVD of data and plot it against q and t axes. n=num of SV to show, smooth=list [x,y] of # of values to median filter by.
    Returns n spectral components, SVs, and time traces.''' 
    if smooth is not None:
        
        if len(data.shape) != len(smooth):
            raise ValueError('smooth must be a list with length = dimensions of data, use 1 for axes you do not wish to smooth.')
        data=medfilt(data,smooth)
    #SVD part
    U, S, V = np.linalg.svd(data, full_matrices=True)   
    spectraces=V[0:n,:]#np.zeros(n,len(qs))
    timetraces=U[:,0:n]#U[0:n,:]#np.zeros(n,len(ts))
    vals=S[0:n]
    
    if showplot:
        #plot 
        if fig is None:
            fig=plt.figure()
            fig.set_size_inches(9, 3)
            a=1
            b=3
            c=1 #set up for using subplots(a,b,c+i)
        else:
            plt.figure(fig)
            #assume sub is 1st of a line, make into (a,b,c)
            if isinstance(sub, str):
                sub=tuple(sub) #format (a,b,c)
            (a,b,c)=sub
            a=int(a)
            b=int(b)
            c=int(c)
            
        #plot S in pretty color points
        Sx=np.arange(1,n+1)
        plt.subplot(a,b,c)
        for i in range(n):
            plt.plot(Sx[i],S[i],'o')
        plt.xlabel('SV')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        #plot time traces
        plt.subplot(a,b,c+1)
        if logscan:
            my_xticks = np.unique(ts)
            ts=np.arange(len(ts))
            my_xticks=['%.4g'%n for n in my_xticks] #make them readable
            plt.xticks(ts, my_xticks,rotation=45)
        plt.plot(ts,timetraces*vals)
        plt.xlabel('t')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        #plot spectral trace
        plt.subplot(a,b,c+2)
        plt.plot(qs,spectraces.T*vals)
        plt.xlabel('Q')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    return spectraces,vals,timetraces


def do_svd_protected(qs,ts,data,n=5,smooth=None,showplot=True, fig=None, sub=None,logscan=False):
    try:
        sp,v,ttr=do_svd(qs,ts,data,n,smooth,showplot,fig,sub,logscan)
    except np.linalg.linalg.LinAlgError:
        nanfilt=~np.isnan(np.nanmean(data,1))
        try:
            sp,v,ttr=do_svd(qs,ts[nanfilt],data[nanfilt],n,smooth,showplot,fig,sub,logscan)
        except np.linalg.linalg.LinAlgError:
            print('SVD failed to converge. Try truncating noisy q edges?')
    return sp,v,ttr
               
    
def Ldiv(a,b,info=False):
    '''Matlab-esque left divide of a vs b. Least-squares fit of a to each column of b, returning coefficients (info=False)
    or more information (residual, rank ,s).'''
    try:
        x,resid,rank,s = np.linalg.lstsq(a,b)
    except np.linalg.LinAlgError:#('1-dimensional array given. Array must be two-dimensional'): one or more vectors is 1d
        try:
            x,resid,rank,s=np.linalg.lstsq(a.reshape(-1,1),b) #maybe the fit vector is 1d
        except np.linalg.LinAlgError:#('1-dimensional array given. Array must be two-dimensional') both 1d! reshape both
            x,resid,rank,s=np.linalg.lstsq(a.reshape(-1,1),b.reshape(-1,1))
    if info:
        return x,resid,rank,s
    else:
        return x