import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import mstats
import pickle as pkl
import sys
import warnings
from scipy.signal import convolve
from scipy.special import legendre
from scipy.integrate import simpson, trapezoid
# from scipy.linalg import lstsq
from scipy import interpolate as inp
from LCLSDataToolsNew.plottingTools import *
from numpy.linalg import inv
from numpy.linalg import lstsq
from sklearn.linear_model import RANSACRegressor as RSC
from sklearn.linear_model import LinearRegression as LR



def theil_sen_stats(y,x):
    '''theil-sen regression returning slope, lower and upper confidence of slope,
    intercept, and lower and upper confidence of the same.'''
    #handle things containing nan with a masked array:
    my = np.ma.masked_array(y, mask=np.isnan(y))
    if (len(my.mask)-sum(my.mask))<3: #if we have less than 3 points not nan, spit out nothing but nan
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #print(sum(my.mask)-len(my.mask))
    
    m,b,lowerm,upperm=mstats.theilslopes(my,x)
    #intercept from median(y) - medslope*median(x)
    #still need error bar on intercepts.
    
    #Old way: always some scaled version of S2 error. 
    #upperb=np.nanmedian(y)-lowerm*np.nanmedian(x)
    #lowerb=np.nanmedian(y)-upperm*np.nanmedian(x)
    
    #new way: standard error of mean for points minus fitted line (spread around the line basically.)
    #95% bands via standard error (x+-1.96*SE)
    nanfilt=~np.isnan(y)
    st_error=np.std(y[nanfilt]-(m*x[nanfilt]+b))/np.sqrt(len(x[nanfilt]))
    
    upperb=b+st_error*1.96
    lowerb=b-st_error*1.96
    return m,lowerm,upperm,b,lowerb,upperb






        
    


def S0S2(dat,phis,fil=None,shift_n=0,deg=None):
    '''Calculate S0 and S2 for each point in dat, using linear fit of I[q] vs P2(cos(phi)). S0, S2 have same length as dat, if filter is used, only filter=True points are filled with data. 
    If a shift was calculated using S0S2_check, use shift_n. 
    deg: input angular shift in degrees.  This will be used instead of shift_n 
    Returns S0, S0 confidence bounds, S2, S2 confidence bounds.'''
    phis=phis[:-1] #hdf5 has one more phi point than slices (they are bin edges)
    S0=np.zeros((dat.shape[0],dat.shape[2]))
    S2=np.zeros((dat.shape[0],dat.shape[2])) #S0 and S2 collapse shape of signal by 1 dimension
    err_S2=np.zeros((dat.shape[0],dat.shape[2],2))
    err_S0=np.zeros((dat.shape[0],dat.shape[2],2))
    
    if deg is not None:
        shift=deg*np.pi/180
        print('shift is %i degrees' %deg) #ie phi zero is at this value
    else:
        shift=phis[shift_n]
        print('shift is %i degrees' %(shift*180/np.pi))
              
    if not (fil is None):
        numerator=np.where(fil)[1] #if a filter is provided, only iterate through interesting points
    else:
        numerator=range(0,dat.shape[0]) #otherwise iterate through all of them. 
    P2_cosphi=(3*np.cos(phis-shift)**2-1)/2 #if we need to rotate, subtract the phi value of the nth phi
    print(len(numerator))
    for pos,i in enumerate(numerator): #for each shot, calculate some stuff
        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()
        cake=dat[i]
        for q in range(0,dat.shape[2]):

            y=cake[:,q]
            m,lowerm,upperm,b,lowerb,upperb=theil_sen_stats(y,P2_cosphi)
            S0[i,q]=b
            S2[i,q]=m
            err_S2[i,q,0]=lowerm
            err_S2[i,q,1]=upperm
            err_S0[i,q,0]=lowerb
            err_S0[i,q,1]=upperb
    return S0, err_S0, S2, err_S2 #want t as first axis second, q as second





def S0S2W(dat,phis,weights=None, shift_n=0,deg=None,thresh=10,DoPrint=True):
    '''Calculate S0 and S2 for each point in dat, using linear fit of I[q] vs P2(cos(phi)). dat has dimensions of phis x qs.
    If a shift was calculated using S0S2_check, use shift_n. 
    deg: input angular shift in degrees.  This will be used instead of shift_n
    Returns S0, S0 standard err, S2, S2 standard err.
    Uses error weights for data, RANSAC regressor for fit.
    thresh=ransac inlier threshold'''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        phis=phis[:-1] #hdf5 has one more phi point than slices (they are bin edges)
        S0=np.zeros((dat.shape[1]))
        S2=np.zeros((dat.shape[1])) #S0 and S2 collapse shape of signal by 1 dimension
        err_S2=np.zeros((dat.shape[1]))
        err_S0=np.zeros((dat.shape[1]))
        MSE_all=np.zeros((dat.shape[1]))

        if deg is not None:
            shift=deg*np.pi/180
            if DoPrint:
                print('shift is %i degrees' %deg)
        else:
            shift=phis[shift_n]
            if DoPrint:
                print('shift is %i degrees' %(shift*180/np.pi))

        P2_cosphi=(3*np.cos(phis-shift)**2-1)/2 #if we need to rotate, subtract the phi value of the nth phi

        #set up RANSAC 
        trialN=10 #number of RANSAC trials to preform
        outRSC=np.full((trialN,2),np.nan)
        reg=RSC(base_estimator=LR(fit_intercept=True), 
                                residual_threshold=thresh,max_trials=1e4,is_data_valid=None)

        for q in range(dat.shape[1]):
            y=dat[:,q]
            if weights is not None:
                cweight=weights[:,q]
            indy=np.nonzero(~np.isnan(y))
            yy=y[indy]
            pp=P2_cosphi[indy]
            ccww=cweight[indy]
            ccww[np.logical_or(np.isnan(ccww),ccww==np.inf)]=0.1
            if yy.shape[0]<=1:
                    continue
            for jj in range(trialN):
                try:
                    #fit_intercept=False: y=mx
                    reg.fit(np.expand_dims(pp,1),yy,sample_weight=ccww)
                    outRSC[jj,0]=reg.estimator_.coef_[0] #slope
                    outRSC[jj,1]=reg.estimator_.intercept_ #intercept 
                    mse = mean_squared_error(reg.predict(np.expand_dims(pp,1)), yy)
                except:
                    outRSC[jj,0]=np.nan
                    outRSC[jj,1]=np.nan
                    mse=np.nan
            #find average slope 
            S2[q]=np.nanmean(outRSC[:,0])
            S0[q]=np.nanmean(outRSC[:,1])
            MSE_all[q]=mse
            err_S0[q]=StandErr(outRSC[:,0])
            err_S0[q]=StandErr(outRSC[:,1])
            
    outfile={'S0':S0,'err_S0':err_S0,'S2':S2,'err_S2':err_S2,'MSE_all':MSE_all,'shift_n':shift_n,'shift_deg':shift}

    return outfile 









def S0S2WT(dat,phis,weights=None, fil=None,shift_n=0,deg=None,thresh=10):
    '''Calculate S0 and S2 for each point in dat, using linear fit of I[q] vs P2(cos(phi)). S0, S2 have same length as dat, if filter is used, only filter=True points are filled with data. 
    If a shift was calculated using S0S2_check, use shift_n. 
    deg: input angular shift in degrees.  This will be used instead of shift_n
    Returns S0, S0 standard err, S2, S2 standard err.
    Uses error weights for data, RANSAC regressor for fit.
    thresh=ransac inlier threshold'''
    
    phis=phis[:-1] #hdf5 has one more phi point than slices (they are bin edges)
    S0=np.zeros((dat.shape[0],dat.shape[2]))
    S2=np.zeros((dat.shape[0],dat.shape[2])) #S0 and S2 collapse shape of signal by 1 dimension
    err_S2=np.zeros((dat.shape[0],dat.shape[2]))
    err_S0=np.zeros((dat.shape[0],dat.shape[2]))
    MSE_all=np.zeros((dat.shape[0],dat.shape[2]))
    if deg is not None:
        shift=deg*np.pi/180
        print('shift is %i degrees' %deg)
    else:
        shift=phis[shift_n]
        print('shift is %i degrees' %(shift*180/np.pi))
        
    P2_cosphi=(3*np.cos(phis-shift)**2-1)/2 #if we need to rotate, subtract the phi value of the nth phi
        
    #filter t bins?   
    if not (fil is None):
        numerator=np.where(fil)[1] #if a filter is provided, only iterate through interesting points
    else:
        numerator=range(0,dat.shape[0]) #otherwise iterate through all of them. 
    print('total time bins', len(numerator))
    
    #set up RANSAC 
    trialN=10 #number of RANSAC trials to preform
    outRSC=np.full((trialN,2),np.nan)
    reg=RSC(base_estimator=LR(fit_intercept=True), 
                            residual_threshold=thresh,max_trials=1e4,is_data_valid=None)
    
    for pos,i in enumerate(numerator): #for each shot, calculate some stuff
        sys.stdout.write('\r'+str(i))
        sys.stdout.flush()

        for q in range(dat.shape[2]):
            y=dat[i,:,q]
            if weights is not None:
                cweight=weights[i,:,q]
            indy=np.nonzero(~np.isnan(y))
            yy=y[indy]
            pp=P2_cosphi[indy]
            ccww=cweight[indy]
            ccww[np.logical_or(np.isnan(ccww),ccww==np.inf)]=0.1
            if yy.shape[0]<=1:
                    continue
            for jj in range(trialN):
                try:
                    #fit_intercept=False: y=mx
                    reg.fit(np.expand_dims(pp,1),yy,sample_weight=ccww)
                    outRSC[jj,0]=reg.estimator_.coef_[0] #slope
                    outRSC[jj,1]=reg.estimator_.intercept_ #intercept 
                    mse = mean_squared_error(reg.predict(np.expand_dims(pp,1)), yy)
                except:
                    return ccww
            #find average slope 
            S2[i,q]=np.nanmean(outRSC[:,0])
            S0[i,q]=np.nanmean(outRSC[:,1])
            MSE_all[i,q]=mse
            err_S0[i,q]=StandErr(outRSC[:,0])
            err_S0[i,q]=StandErr(outRSC[:,1])
            
    outfile={'S0':S0,'err_S0':err_S0,'S2':S2,'err_S2':err_S2,'MSE_all':MSE_all,'shift_n':shift_n,'shift_deg':shift}
    #want t as first axis second, q as second

    return outfile 





              
def S0S2_check(dat,qs,phis,trange='All',lim=None,calc2=False):
    '''Average the data IN TIME and plot the S0 and S2, with uncertainties from T-S estimator, 
    with phi0 rotating around the full circle. 
    Checks whether phi=0 is in the right place to see maximum anisotropic signal.
    trange= array of time bins to average
    lim= sets y limits of S2 plots'''
    
    if trange=='All':
        cake=np.nanmean(dat,0)
    else:
        cake=np.nanmean(dat[trange,:,:],0)
                        
     
                        
    phis=phis[:-1] #hdf5 has one more phi point than slices (they are bin edges)
    ylow=0
    yhigh=0
    
    all_S0=np.zeros((len(phis),len(qs)))
    all_S2=np.zeros((len(phis),len(qs)))
   
    shifts=range(len(phis))                
    for i in range(len(phis)):
        shift_n=i
#         shiftval=(2*np.pi)/cake.shape[0] #phi width of slices of cake

        P2_cosphi=(3*np.cos(phis-phis[shift_n])**2-1)/2
        S0=np.zeros(cake.shape[1])
        S2=np.zeros(cake.shape[1])
        conf_m=np.zeros((cake.shape[1],2))
        conf_b=np.zeros((cake.shape[1],2))
        for q in range(0,cake.shape[1]):
            y=cake[:,q]
            m,lowerm,upperm,b,lowerb,upperb=theil_sen_stats(y,P2_cosphi)
            S0[q]=b
            S2[q]=m
            conf_m[q,0]=lowerm
            conf_m[q,1]=upperm
            conf_b[q,0]=lowerb
            conf_b[q,1]=upperb
            
        all_S0[i,:]=S0
        all_S2[i,:]=S2
        
#         plt.figure('shift %i'%i)
        #plt.title('shift %i'%i)
        #plt.plot(qs,S0,'bo')
        #plt.plot(S2,'ro')
        
        
        plt.figure('all S0 trace')
        plt.errorbar(qs,S0, yerr=[S0-conf_b[:,0],conf_b[:,1]-S0], fmt='--o',label='shift %i'%i)
        plt.legend()
        plt.xlabel('Q ($\AA^{-1}$)')
        plt.ylabel('I (arb)')

        plt.figure('all S2 trace')
        plt.errorbar(qs,S2, yerr=[S2-conf_m[:,0],conf_m[:,1]-S2], fmt='--o',label='shift %i'%i)
        plt.legend()
        plt.xlabel('Q ($\AA^{-1}$)')
        plt.ylabel('I (arb)')
        

        ax=plt.gca()
        yl,yh=ax.get_ylim()
        if yh>yhigh:
            yhigh=yh
        if yl<ylow:
            ylow=yl
            
    
    plt.figure('all S2 trace')
    if lim is None:
        plt.ylim(ylow*0.5,yhigh*0.5)
    else:
        plt.ylim(lim)
            

            
    plot_2d(shifts,qs,all_S0,fig='all S0',cb=True)
    plt.ylabel('q')
    plt.xlabel('phi shift')
    plt.suptitle('all S0')
    plt.xticks(np.arange(len(shifts))+1)
    
    plot_2d(shifts,qs,all_S2,fig='all S2',cb=True)
    plt.ylabel('q')
    plt.xlabel('phi shift')
    plt.suptitle('all S2')
    plt.xticks(np.arange(len(shifts))+1)
    
    
    return all_S0, all_S2
              
              
def S0S2_Fine(dat,qs,phis,shiftLim=(0,360),step=10,trange='All',lim=None):
    '''Average the data IN TIME and plot the S0 and S2, with uncertainties from T-S estimator, 
    with phi0 rotating around the full circle. (ie, steps are user deterined, not equal to number of phi bins)
    Checks whether phi=0 is in the right place to see maximum anisotropic signal.
    shiftLim=range of angles to scan, in degrees
    step=step size of angle scan, in degrees
    trange= array of time bins to average
    lim= sets y limits of S2 plots'''
    
    if trange=='All':
        cake=np.nanmean(dat,0)
    else:
        cake=np.nanmean(dat[trange,:,:],0)
                        
     
                        
    phis=phis[:-1] #hdf5 has one more phi point than slices (they are bin edges)
    ylow=0
    yhigh=0
    

    
    shifts=np.arange(shiftLim[0]*np.pi/180,shiftLim[1]*np.pi/180,(step*np.pi/180))
    all_S0=np.zeros((len(shifts),len(qs)))
    all_S2=np.zeros((len(shifts),len(qs)))
   
                    
    for i,ss in enumerate(shifts):
        P2_cosphi=(3*np.cos(phis-ss)**2-1)/2
        S0=np.zeros(cake.shape[1])
        S2=np.zeros(cake.shape[1])
        conf_m=np.zeros((cake.shape[1],2))
        conf_b=np.zeros((cake.shape[1],2))
        for q in range(0,cake.shape[1]):
            y=cake[:,q]
            m,lowerm,upperm,b,lowerb,upperb=theil_sen_stats(y,P2_cosphi)
            S0[q]=b
            S2[q]=m
            conf_m[q,0]=lowerm
            conf_m[q,1]=upperm
            conf_b[q,0]=lowerb
            conf_b[q,1]=upperb
            
        all_S0[i,:]=S0
        all_S2[i,:]=S2
        
#         plt.figure('shift %i'%i)
        #plt.title('shift %i'%i)
        #plt.plot(qs,S0,'bo')
        #plt.plot(S2,'ro')
        
        
        
        
        plt.figure('all S0 trace')
        plt.errorbar(qs,S0, yerr=[S0-conf_b[:,0],conf_b[:,1]-S0], fmt='--o',label='shift %i rad'%i)
        plt.legend()
        plt.xlabel('Q ($\AA^{-1}$)')
        plt.ylabel('I (arb)')

        plt.figure('all S2 trace')
        plt.errorbar(qs,S2, yerr=[S2-conf_m[:,0],conf_m[:,1]-S2], fmt='--o',label='shift %i rad'%i)
        plt.legend()
        plt.xlabel('Q ($\AA^{-1}$)')
        plt.ylabel('I (arb)')
        

        ax=plt.gca()
        yl,yh=ax.get_ylim()
        if yh>yhigh:
            yhigh=yh
        if yl<ylow:
            ylow=yl
            
    
    plt.figure('all S2 trace')
    if lim is None:
        plt.ylim(ylow*0.5,yhigh*0.5)
    else:
        plt.ylim(lim)
            


    shifts=shifts*180/np.pi
    print(shifts.shape)
    plot_2d(shifts,qs,all_S0,fig='all S0',cb=True)
    plt.ylabel('q')
    plt.xlabel('phi shift (degrees)')
    plt.suptitle('all S0')
#     plt.xticks(np.arange(len(shifts))+1)
    
    plot_2d(shifts,qs,all_S2,fig='all S2',cb=True)
    plt.ylabel('q')
    plt.xlabel('phi shift (degrees)')
    plt.suptitle('all S2')
#     plt.xticks(np.arange(len(shifts))+1)
    
    
    return all_S0, all_S2
              
              
              
              
              
              
              
              
              
              
              
              
              
              


def S0S2Int(data,qs,phisIn,shift_n=0):
    '''Based on Michael Chen's method. where data has dimensions t x phi x qs. 
    Integrate from 0 to pi'''
    
    phis=phisIn.copy()
#     phis+=2*np.pi/phis.shape[0]*shift_n #shift phis accordingly
    phis-=phis[shift_n] 
    
    #shift values greater than 2pi and resort wrt phi
    phis=np.mod(phis,2*np.pi)
    SortIn=np.argsort(phis)
    phis=phis[SortIn]
    data=data[:,SortIn,:]
    
    #isolate phis from 0 to pi
    index=np.nonzero(np.logical_and(phis>=0,phis<np.pi))
    phis1=phis[index]
    data1=np.squeeze(data[:,index,:])
    
    leg0 = legendre(0)
    leg2 = legendre(2)
    s0_vals=np.zeros((data.shape[0],data.shape[2])) #tbins x qbins
    s2_vals=np.zeros((data.shape[0],data.shape[2]))
    for i in range(data.shape[0]):
        for j,qq in enumerate(qs):
            #remove nan
            data2=data1[i,:,j].copy()
            index2=np.nonzero(~np.isnan(data2))
            data2=data2[index2]
            phis2=phis1[index2]
            

            s0_vals[i,j] = 1.0/2.0*trapezoid(data2*leg0(np.cos(phis2))*np.sin(phis2), x=phis2)
            s2_vals[i,j] = 5.0/2.0*trapezoid(data2*leg2(np.cos(phis2))*np.sin(phis2), x=phis2)

    outfile={'S0':s0_vals,'S2':s2_vals,'shift_n': shift_n}
    return outfile


def SnInt(nn,data,qs,phisIn,shift_n=0,method='trap'):
    '''Based on Michael Chen's integration method, calculate nn-th Legendre component.
    Data has dimensions phixqs, NO TIME DIMENSION, integrate from 0 to pi. shift_n=shift in phi bins.'''
    phis=phisIn.copy()
    #     phis+=2*np.pi/phis.shape[0]*shift_n #shift phis accordingly
    shift=phis[shift_n]
    phis-=shift 
    print('shift is %i degrees' %(shift*180/np.pi)) 
    
    #shift values greater than 2pi and resort wrt phi
    phis=np.mod(phis,2*np.pi)
    SortIn=np.argsort(phis)
    phis=phis[SortIn]
    data=data[SortIn,:]
    
    #isolate phis from 0 to pi
    index=np.nonzero(np.logical_and(phis>=0,phis<np.pi))
    phis1=phis[index]
    data1=np.squeeze(data[index,:])

    legn = legendre(nn)
    sn_vals=np.full((data.shape[1]),np.nan) # qbins

    for j,qq in enumerate(qs):
        #remove nan
        data2=data1[:,j].copy()
        index2=np.nonzero(~np.isnan(data2))
        data2=data2[index2]
        phis2=phis1[index2]
        
        try:
            if method=='trap':
                sn_vals[j]=(2*nn+1)/2.0*trapezoid(data2*legn(np.cos(phis2))*np.sin(phis2), x=phis2)
            elif method=='simpson':
                sn_vals[j]=(2*nn+1)/2.0*simpson(data2*legn(np.cos(phis2))*np.sin(phis2), x=phis2)
        except:
            continue

    
    outfile={'n': nn, 'Sn_vals':sn_vals, 'shift_n': shift_n}
    return outfile 

def SnIntT(nn,data,qs,phisIn,shift_n=0,method='trap'):
    '''Based on Michael Chen's integration method, calculate nn-th Legendre component.
    Data has dimensions txphixqs, integrate from 0 to pi. shift_n=shift in phi bins.'''
    phis=phisIn.copy()
    #     phis+=2*np.pi/phis.shape[0]*shift_n #shift phis accordingly
    shift=phis[shift_n]
    phis-=shift 
    print('shift is %i degrees' %(shift*180/np.pi)) 
    
    #shift values greater than 2pi and resort wrt phi
    phis=np.mod(phis,2*np.pi)
    SortIn=np.argsort(phis)
    phis=phis[SortIn]
    data=data[:,SortIn,:]
    
    #isolate phis from 0 to pi
    index=np.nonzero(np.logical_and(phis>=0,phis<np.pi))
    phis1=phis[index]
    data1=np.squeeze(data[:,index,:])

    legn = legendre(nn)
    sn_vals=np.full((data.shape[0],data.shape[2]),np.nan) #tbins x qbins
    
    for i in range(data.shape[0]):
        for j,qq in enumerate(qs):
            #remove nan
            data2=data1[i,:,j].copy()
            index2=np.nonzero(~np.isnan(data2))
            data2=data2[index2]
            phis2=phis1[index2]

            try:
                if method=='trap':
                    sn_vals[i,j]=(2*nn+1)/2.0*trapezoid(data2*legn(np.cos(phis2))*np.sin(phis2), x=phis2)
                elif method=='simpson':
                    sn_vals[i,j]=(2*nn+1)/2.0*simpson(data2*legn(np.cos(phis2))*np.sin(phis2), x=phis2)
            except:
                continue
    
    outfile={'n': nn, 'Sn_vals':sn_vals, 'shift_n': shift_n}
    return outfile


def Sn_check(nn,dat,qs,phis,trange='All',lim=None,shifts='All'):
    '''Average the data IN TIME and plot the S0 and S2, with uncertainties from T-S estimator, 
    with phi0 rotating around the full circle. 
    Using Michael Chen integration method. 
    Checks whether phi=0 is in the right place to see maximum anisotropic signal.
    trange= array of time bins to average
    lim= sets y limits of S2 plots
    shifts='All' try all shift_n 
    shifts=list-like :  try shift_n in list'''
    
    if trange=='All':
        cake=np.nanmean(dat,0)
    else:
        cake=np.nanmean(dat[trange,:,:],0).squeeze()
    data=cake
                        
    phis=phis[:-1] #hdf5 has one more phi point than slices (they are bin edges)
    ylow=0
    yhigh=0
    
    
    
    if shifts=='All':
        all_Sn=np.zeros((len(phis),len(qs)))
        shifts=range(len(phis))
    else:
        all_Sn=np.zeros((len(shifts),len(qs)))
        
    colors=mpl.cm.rainbow(np.linspace(0,1,len(shifts))) 
    for ss,ii in enumerate(shifts):
        shift_n=ii
  
        phisT=phis.copy()
        phisT-=phisT[shift_n] 
        
        #shift values greater than 2pi and resort wrt phi
        phisT=np.mod(phisT,2*np.pi)
        SortIn=np.argsort(phisT)
        phisT=phisT[SortIn]
        dataT=data[SortIn,:]

        #isolate phis from 0 to pi
        index=np.nonzero(np.logical_and(phisT>=0,phisT<np.pi))
        phis1=phisT[index].squeeze()
        data1=np.squeeze(dataT[index,:])

        legn = legendre(nn)
        sn_vals=np.zeros((data.shape[1])) #tbins x qbins

        for j,qq in enumerate(qs):
            #remove nan
            data2=data1[:,j].copy()
            index2=np.nonzero(~np.isnan(data2))
            data2=data2[index2]
            phis2=phis1[index2]

            sn_vals[j]=(2*nn+1)/2.0*trapezoid(data2*legn(np.cos(phis2))*np.sin(phis2), x=phis2)
        
        all_Sn[ss,:]=sn_vals

        plt.figure('all Sn trace')
        plt.plot(qs,sn_vals,label='shift %i'%ii,c=colors[ss])
        plt.legend()
        plt.xlabel('Q ($\AA^{-1}$)')
        plt.ylabel('I (arb)')
        

        ax=plt.gca()
        yl,yh=ax.get_ylim()
        if yh>yhigh:
            yhigh=yh
        if yl<ylow:
            ylow=yl
            
    
    plt.figure('all Sn trace')
    if lim is None:
        plt.ylim(ylow*0.5,yhigh*0.5)
    else:
        plt.ylim(lim)
            

    plot_2d(shifts,qs,all_Sn,fig='all Sn',cb=True)
    plt.ylabel('q')
    plt.xlabel('phi shift')
    plt.suptitle('all S_%i'%nn)
    xaxis=np.arange(np.min(shifts)-1,np.max(shifts)+2)
    plt.xticks(xaxis)
    
    return all_Sn

              

              
              
              
def SnFit(nn,data,qs,phisIn,lam,shift_n=0):
    '''based on Adi Natan method (see https://github.com/adinatan/AnalyzeScatteringSignal/blob/master/LDSD.m, 
    DOI: 10.1039/D0FD00126K)
       for equally spaced q and phi bins (ie same number of phi bins for each q),
       calculates Sn. By least squares fit of system of equations:
           Data=S0*P0(cos(phi))+...+Sn*Pn(cos(phi))
       nn=int fits all even orders up to nn (ie nn=2 calculates S0,S2).
       if nn=list then fits only orders specified
       lam= input x-ray wavelength in angstroms
       Data has dimensions phis x qs.
       b_All=dimensionless, scaled beta values (B0*b_All=beta values)
       b_Res=sums of squared residuals'''
    
    #shift phis
    phis=phisIn.copy()
    shift=phis[shift_n]
    phis-=shift 
    print('shift is %i degrees' %(shift*180/np.pi))
    
    #shift values greater than 2pi and resort wrt phi
    phis=np.mod(phis,2*np.pi)
    SortIn=np.argsort(phis)
    phis=phis[SortIn]
    data=data[SortIn,:]
    
    #even legendre orders to be included
    if type(nn)== int:
        legOrd=np.arange(0,nn+1,2)
    else:
        legOrd=np.array(nn)

    #find B0(q)
    B0=np.nanmean(data,0)
    dNorm=data/B0

    #initialize output
    b_All=np.full((qs.shape[0],legOrd.shape[0]),np.nan)
    b_Res=np.full((qs.shape[0]),np.nan)
    
    #for each q
    for ii,qq in enumerate(qs):
        #initialize arrays
        Amat=np.zeros((phis.shape[0],legOrd.shape[0]))
        Ytemp=dNorm[:,ii]
        
        #get rid of nan values
        Ind2=np.nonzero(~np.isnan(Ytemp))
        YY=Ytemp[Ind2]
        
        #write function
        for jj,lN in enumerate(legOrd):
            legn=legendre(lN)
            Amat[:,jj]=legn(np.cos(phis))
#         b_coeff=inv(Amat.T@Amat)@Amat.T@Ytemp #manually solve
        AA=Amat[Ind2,:].squeeze()    
        b_coeff,b_res1= lstsq(AA,YY)[:2]
        b_All[ii,:]=b_coeff
        try:
            b_Res[ii]=b_res1
        except:
            continue
        
    Sn=np.full_like(b_All,np.nan)
    for kk,nn in enumerate(legOrd):
        for ii,qq in enumerate(qs):
            Sn[ii,kk]=B0[ii]*b_All[ii,kk]*(1-qq**2/(4*(2*np.pi/lam)**2))**(-nn/2)
            
    Outfile={'b_All':b_All,'B0':B0,'legOrd':legOrd,'Sn':Sn,'shift_n':shift_n,'lambda':lam,'b_Res':b_Res}
    
    return Outfile


def SnFitT(nn,data1,qs,phisIn,lam,shift_n=0):
    '''SnFit but with data that has dimensions txphixq, returns txqxnn array'''

    #shift phis
    phis=phisIn.copy()
    shift=phis[shift_n]
    phis-=shift 
    print('shift is %i degrees' %(shift*180/np.pi)) 

    #shift values greater than 2pi and resort wrt phi
    phis=np.mod(phis,2*np.pi)
    SortIn=np.argsort(phis)
    phis=phis[SortIn]

    #even legendre orders to be included
    if type(nn)== int:
        legOrd=np.arange(0,nn+1,2)
    else:
        legOrd=np.array(nn)

    Output=np.full((data1.shape[0],data1.shape[2],legOrd.shape[0]),np.nan)
    B0_A=np.full((data1.shape[0],data1.shape[2]),np.nan)
    b_Res=np.full((data1.shape[0],data1.shape[2]),np.nan)
    Out2=[]
    for kk in range(data1.shape[0]):

        data=data1[kk,SortIn,:].copy()

        #find B0(q)
        B0=np.nanmean(data,0)
        dNorm=data/B0

        #initialize output
        b_All=np.full((qs.shape[0],legOrd.shape[0]),np.nan)

        
        #for each q
        for ii,qq in enumerate(qs):
            #initialize arrays
            Amat=np.zeros((phis.shape[0],legOrd.shape[0]))
            Ytemp=dNorm[:,ii]

            #get rid of nan values
            Ind2=np.nonzero(~np.isnan(Ytemp))
            YY=Ytemp[Ind2]

            #write function
            for jj,lN in enumerate(legOrd):
                legn=legendre(lN)
                Amat[:,jj]=legn(np.cos(phis))
    #         b_coeff=inv(Amat.T@Amat)@Amat.T@Ytemp #manually solve
            AA=Amat[Ind2,:].squeeze()    
            b_coeff,b_res1= lstsq(AA,YY)[:2]
            b_All[ii,:]=b_coeff
            try:
                b_Res[kk,ii,:]=b_res1
            except:
                continue
            

        Output[kk,:,:]=b_All
        B0_A[kk,:]=B0

        
#     #list of lists to array, first make all equal length
#     length=max(map(len, b_Res))
#     b_Res=np.array([xi+[np.nan]*(length-len(xi)) for xi in b_Res])
    
    Sn=np.full_like(Output,np.nan)
    for kk,nn in enumerate(legOrd):
        for ii,qq in enumerate(qs):
            Sn[:,ii,kk]=B0_A[:,ii]*Output[:,ii,kk]*(1-qq**2/(4*(2*np.pi/lam)**2))**(-nn/2)
            
    Outfile={'b_All':Output,'B0_All':B0_A,'legOrd':legOrd,'Sn':Sn,'shift_n':shift_n,'lambda':lam, 'b_Res':b_Res}
    
    return Outfile
            
       
        
        

def StandErr(A,axis=None):
    ''' for array A calculate standard error. Calculates along axis=(0,1)etc.'''
    size=0
    if axis == None:
         B=np.nanstd(A)/np.sqrt(len(A))
    else:
        if type(axis) is not int:
            for i in axis:
                size+=A.shape[i]
        else:
            size=A.shape[axis]
        B=np.nanstd(A,axis=axis)/np.sqrt(size)
    return B