import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from scipy import interpolate as inp
import warnings


def ReBin(qs,image,binSize=0.01,binNum=None,binRange=None):
    ''' image with dimensions tbins x phibins x qbins is rebinned in q using the given bin size.  
    if binNum is given then that number of points per a bin is used instead.  Returns new qs and new image.
    If binRange=(start,end) then use those end points for new q binning.  Otherwise use range of qs.''' 
    
    if binNum is not None:
        edges,centres=MakeEqualBins(qs,binNum)
    elif binRange is not None:
        low=binRange[0]
        high=binRange[1]
        edges=np.arange(low,high,binSize)
        centres=edges[0:-1]+binSize/2
    else:
        low=np.nanmin(qs)
        high=np.nanmax(qs)+binSize
        edges=np.arange(low,high,binSize)
        centres=edges[0:-1]+binSize/2
    
    outfile ={}
    outfile['old_qRange']=(np.nanmin(qs),np.nanmax(qs))
    outfile['new_qs']=centres
    
    if np.any(image==np.nan):
        print('NaN values!  Interpolation may be unreliable')
        
    inpFn=inp.interp1d(qs,image,axis=-1, kind='linear', bounds_error=False, fill_value=np.nan)
    
    outfile['new_im']=inpFn(centres) 
    
    return outfile
 
    
    
def ReBinOLD(qs,image,binSize=0.01,binNum=None,binRange=None):
    ''' image with dimensions tbins x phibins x qbins is rebinned in q using the given bin size.  
    if binNum is given then that number of points per a bin is used instead.  Returns new qs and new image.
    If binRange=(start,end) then use those end points for new q binning.  Otherwise use range of qs.''' 
    
    if binNum is not None:
        edges,centres=MakeEqualBins(qs,binNum)
    elif binRange is not None:
        low=binRange[0]
        high=binRange[1]
        edges=np.arange(low,high,binSize)
        centres=edges[0:-1]+binSize/2
    else:
        low=np.nanmin(qs)
        high=np.nanmax(qs)+binSize
        edges=np.arange(low,high,binSize)
        centres=edges[0:-1]+binSize/2
    
    outfile ={}
    outfile['new_qs']=centres   
    
    #initialize
    if len(image.shape)==2:
        image=np.expand_dims(image,1)
    phi_len=image.shape[1]
    t_len=image.shape[0]
    bin_len=edges.size-1
    binmean=np.zeros((phi_len,bin_len,t_len)) #initialize array of size phi, qs, ts
    binstd=np.zeros((phi_len,bin_len,t_len))

    #bin in phi
    for i in range(image.shape[1]):
        im_T=image[:,i,:].T #transpose to be qs x tbins to use bin in q code and cheat
        out_temp=BinInQ(qs,im_T,edges,yerr=None)
        binmean[i]=out_temp['binmean']
    #         binstd[i]=out_temp['binstd'] 

    #store outputs with array dimensions bins,phis, q
    outfile['new_im']=np.transpose(binmean,(2,0,1))
    #     outfile['binstd']=np.transpose(binstd,(2,0,1))

    return outfile

def BinStat(x,y,n,yerr=None,binByPoints=True,showplot=False,set_bins=None,count=True, xstat=True):
    ''' bin for a data set y with dimensions = (shots,phi,q)
        returns mean and sd for each bin as a dictionary type output file
        yerr is standard deviation wrt q bins, error is propagated and new standard deviation is returned.
        if count=True also returns number of shots per bin
        if xstat=True also returns mean and std wrt x for each bin
         If binByPoints=True, n=number of points per bin. If n=0, numpy estimates good bin number. Divide n=total points/#bins.
    if binByPoints=False, n=number of bins. If n=0, numpy will estimate a good bin number ('auto' method of np.histogram).'''

        
    outfile={}
    
    warnings.catch_warnings()
    warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
    warnings.simplefilter("ignore", category=FutureWarning)
    
    #Calculate bin edges
    if set_bins is not None:
        edges=set_bins
        centres=edges[0:-1]+np.diff(edges)/2
    elif binByPoints:
        if n>0:
            edges,centres=MakeEqualBins(x,n)
        else:
            fill,edges=np.histogram(x,'auto')
            n=len(x)/len(edges) #make that many bins, but do it by points. 
            edges,centres=MakeEqualBins(x,n)
    else:
        if n>0:
            fill,edges=np.histogram(x,n)
        else:
            fill,edges=np.histogram(x,'auto')
        centres=edges[0:-1]+np.diff(edges)/2

    outfile['xcenter']=centres

    if xstat:
        xmean=binned_statistic(x,x,np.nanmean,bins=edges)[0]
        xstd=binned_statistic(x,x,np.nanstd,bins=edges)[0]
        outfile['xmean']=xmean
        outfile['xstd']=xstd

    if count:
        bincount=binned_statistic(x,x,'count',bins=edges)[0] 
        outfile['bincount']=bincount


    #are there phi cakes?
    if len(y.shape)==3:

        phi_len=y[1,:,1].size
        q_len=y[1,1,:].size
        bin_len=edges.size-1
        binmean=np.zeros((phi_len,bin_len,q_len)) #initialize array of size phi,bins,q
        binstd=np.zeros((phi_len,bin_len,q_len))
        if yerr is not None:
            qerr=np.zeros((phi_len,bin_len,q_len))
            # for each phi: 
            for i in range(phi_len):
                out_temp=BinInQ(x,y[:,i,:],edges,yerr=yerr[:,i,:])
                binmean[i]=out_temp['binmean']
                binstd[i]=out_temp['binstd']
                qerr[i]=out_temp['qerr']
            #store outputs with array dimensions bins,phis, q
            outfile['binmean']=np.transpose(binmean,(1,0,2))
            outfile['binstd']=np.transpose(binstd,(1,0,2))
            outfile['qerr']=np.transpose(qerr,(1,0,2))
        else:
            # for each phi: 
            for i in range(phi_len):
                out_temp=BinInQ(x,y[:,i,:],edges,yerr=None)
                binmean[i]=out_temp['binmean']
                binstd[i]=out_temp['binstd']
                # print(i, 'Done')
            #store outputs with array dimensions bins,phis, q
            outfile['binmean']=np.transpose(binmean,(1,0,2))
            outfile['binstd']=np.transpose(binstd,(1,0,2))

    else:
        outfile.update(BinInQ(x,y,edges,yerr=yerr))

    return outfile


        
        
        
def BinStatbyAxis(x,y,n,yerr=None,binByPoints=True,showplot=False,set_bins=None,count=True,xstat=True):
    ''' Bin centers are determined by unique values of x!
        bin for a data set y with dimensions = (shots,phi,q)
        returns mean and sd for each bin as a dictionary type output file
        if count=True also returns number of shots per bin'''


    outfile={}
    
    warnings.catch_warnings()
    warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
    warnings.simplefilter("ignore", category=FutureWarning)
    
    #Calculate bin edges
    centres = np.unique(x)

    shift=np.append(np.diff(centres)[0]/2,np.diff(centres)/2)
    edges=centres-shift
    edges=np.append(edges,centres[-1]+np.diff(centres)[-1])

    outfile['xcenter']=centres
    
    if xstat:
        xmean=binned_statistic(x,x,np.nanmean,bins=edges)[0]
        xstd=binned_statistic(x,x,np.nanstd,bins=edges)[0]
        outfile['xmean']=xmean
        outfile['xstd']=xstd
    
    if count:
        bincount=binned_statistic(x,x,'count',bins=edges)[0] 
        outfile['bincount']=bincount

    #are there phi cakes?
    if len(y.shape)==3:

        phi_len=y[1,:,1].size
        q_len=y[1,1,:].size
        bin_len=edges.size-1
        binmean=np.zeros((phi_len,bin_len,q_len)) #initialize array of size phi,bins,q
        binstd=np.zeros((phi_len,bin_len,q_len))
        if yerr is not None:
            qerr=np.zeros((phi_len,bin_len,q_len))
            # for each phi: 
            for i in range(phi_len):
                out_temp=BinInQ(x,y[:,i,:],edges,yerr=yerr[:,i,:])
                binmean[i]=out_temp['binmean']
                binstd[i]=out_temp['binstd']
                qerr[i]=out_temp['qerr']
            #store outputs with array dimensions bins,phis, q
            outfile['binmean']=np.transpose(binmean,(1,0,2))
            outfile['binstd']=np.transpose(binstd,(1,0,2))
            outfile['qerr']=np.transpose(qerr,(1,0,2))
        else:
            # for each phi: 
            for i in range(phi_len):
                out_temp=BinInQ(x,y[:,i,:],edges,yerr=None)
                binmean[i]=out_temp['binmean']
                binstd[i]=out_temp['binstd']
            #store outputs with array dimensions bins,phis, q
            outfile['binmean']=np.transpose(binmean,(1,0,2))
            outfile['binstd']=np.transpose(binstd,(1,0,2))

    else:
        outfile.update(BinInQ(x,y,edges,yerr=yerr))
    return outfile
        

  
        
def BinInQ(x,y,edges,yerr=None):
    ''' for y with dimensions (shots, q), calculate bin mean and variance for x bins given by edges, propagates error yerr=std'''
       
    outfile={}
    
    warnings.catch_warnings()
    warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
    warnings.simplefilter("ignore", category=FutureWarning)
    
    
    binmean=np.full((edges.shape[0]-1,y.shape[1]),np.nan)
    binstd=np.full((edges.shape[0]-1,y.shape[1]),np.nan)
    
    for i,ed in enumerate(edges[:-1]):
        index=np.nonzero(np.logical_and(x>=ed,x<edges[i+1]))
        if index[0].size==0:
#             print('empty bin')
            continue
        t1=y[index,:].squeeze()
        if t1.shape==1:
            t1=np.expand_dims(t1,0)
        try:
            ave1=np.nanmean(t1,0)
#             std1=np.nanstd(t1,0)
            std1=StandErr(t1,0)
            
            binmean[i,:]=ave1
            binstd[i,:]=std1
        except:
            print('binning error')
            continue
    
    outfile['binmean']=binmean
    outfile['binstd']=binstd 
                                        
    if yerr is not None:
#         print('doing yerr')
        yerrL=yerr.T.tolist()                                  
        binyerr=binned_statistic(x,yerrL,ErrorProp,bins=edges)[0]
        outfile['qerr']=np.transpose(binyerr)
    
    return outfile
    









def ErrProp(B,axis=None):
    ''' for Nd array a1,a2,a3 that that are added or subtracted, propagate the standard deviations given in B 
    where b1 is error for a1. Calculates along axis=(0,1)etc. '''
    C=np.sqrt(np.nansum(B**2,axis=axis))
    return C



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










def MakeEqualBins(x,n,showplot=False):
    '''partition x axis into bins such that each bin contains n points.
    Returns edges and centres of bins.'''
    sortx=np.sort(x[~np.isnan(x)])
    #print(sortx[:30])
    num_bins=round(len(sortx)/n)
    #print('%i bins'%num_bins)
    edges=np.zeros(num_bins+1)
    edges[0]=np.nanmin(sortx)
    edges[-1]=np.nanmax(sortx)
    for ii in range(1,int(num_bins)):
        edges[ii]=sortx[ii*n]
    if showplot:
        plt.figure()
        plt.xlabel('x')
        plt.title('Histogram: %d bins; %d events in each' %(num_bins,n))
        plt.hist(x,200)
        for ii in range(len(edges)):
            plt.axvline(edges[ii])
    centres=edges[0:-1]+np.diff(edges)/2
    #print(edges)
    return edges, centres

def BinnedMean(x,y,n,binByPoints=True,showplot=False,set_bins=None,count=False):
    '''Bin along x, then average the y points in each bin.
    Returns center of each bin and the mean y value in that bin.
    If binByPoints=True, n=number of points per bin. If n=0, numpy estimates good bin number. Divide n=total points/#bins.
    if binByPoints=False, n=number of bins. If n=0, numpy will estimate a good bin number ('auto' method of np.histogram).
    if set_bins are given as a list of bin edges, these will be used, overriding other options. ''' 
#     print(y.shape)
    if set_bins is not None:
        edges=set_bins
        centres=edges[0:-1]+np.diff(edges)/2
    elif binByPoints:
        if n>0:
            edges,centres=MakeEqualBins(x,n)
        else:
            fill,edges=np.histogram(x,'auto')
            n=len(x)/len(edges) #make that many bins, but do it by points. 
            edges,centres=MakeEqualBins(x,n)
    else:
        if n>0:
            fill,edges=np.histogram(x,n)
        else:
            fill,edges=np.histogram(x,'auto')
        centres=edges[0:-1]+np.diff(edges)/2
        
    
    if x.shape==y.shape:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
            binmeans = binned_statistic(x,y,np.nanmean,bins=edges)[0] #scipy.stats.binned_statistic
            bincount=binned_statistic(x,y,'count',bins=edges)[0]
    else:
        #need to give binned_statistic a list of lists for it to be happy and compute many means
        #need to filter out inf to nan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
            yL=y.T.tolist()
            binmeans = binned_statistic(x,yL,np.nanmean,bins=edges)[0]
    if count:
        bincount=binned_statistic(x,yL,'count',bins=edges)[0]
        if showplot:
#             plt.figure()
#             plt.hist(bincount,bins=edges)
            
            plt.figure()      
            plt.plot(x,y,'b.')
            plt.plot(centres,binmeans.T,'ro')
            legend_elements = [Line2D([0], [0], color='b', marker='.', label='all points'),
                   Line2D([0], [0], marker='o', color='r', label='mean vs center')]
            plt.legend(handles=legend_elements)
            plt.xlabel('t')
        return centres, binmeans, bincount
    else:
        if showplot:
            plt.figure()      
            plt.plot(x,y,'b.')
            plt.plot(centres,binmeans.T,'ro')
            legend_elements = [Line2D([0], [0], color='b', marker='.', label='all points'),
                   Line2D([0], [0], marker='o', color='r', label='mean vs center')]
            plt.legend(handles=legend_elements)
            plt.xlabel('t')
            
        return centres, binmeans


def BinnedMeanCake(x,y,n,binByPoints=True,showplot=False,set_bins=None,count=False):
    print('yshape', y.shape)
    '''Bin along x, then average the y points in each bin. Repeat along axis 1 (phi axis) to create binned mean at each phi.
    Returns center of each bin and the mean y value in that bin.
    If binByPoints=True, n=number of points per bin.
    if binByPoints=False, n=number of bins. If n=0, numpy will estimate a good bin number ('auto' method of np.histogram).'''
    ts,means=BinnedMean(x,y[:,0,:],n,binByPoints,set_bins=set_bins)
    binnd=np.zeros((means.shape[1],y.shape[1],y.shape[2]))       
    for i in range(y.shape[1]): #for each phi slice, do the timebin
        ts,means=BinnedMean(x,y[:,i,:],n,binByPoints,showplot,set_bins)
        binnd[:,i,:]=means.T
#         print('binning:', i)
    if count: #same number per bin in any slice. only count once.
        ts,means,occ=BinnedMean(x,y[:,0,:],n,binByPoints,showplot,set_bins,count=True)
        return ts,binnd,occ[0,:]
    
    return ts,binnd


def BinnedMeanByAxis(x,y,showplot=False,count=False):
    '''Bin along x, then average the y points in each bin.
    Returns center of each bin and the mean y value in that bin.
    Bin by the unique values in x.''' 
    centres = np.unique(x)
    
    shift=np.append(np.diff(centres)[0]/2,np.diff(centres)/2)
    edges=centres-shift
    edges=np.append(edges,centres[-1]+np.diff(centres)[-1])
    if x.shape==y.shape:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
            binmeans = binned_statistic(x,y,np.nanmean,bins=edges)[0] #scipy.stats.binned_statistic
            if count:
                occ=binned_statistic(x,y,'count',bins=edges)[0]
    else:
        #need to give binned_statistic a list of lists for it to be happy and compute many means
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
            binmeans = binned_statistic(x,y.T.tolist(),np.nanmean,bins=edges)[0]
            if count:
                occ=binned_statistic(x,y,'count',bins=edges)[0]
    if showplot:
        plt.figure()      
        plt.plot(x,y,'b.')
        plt.plot(centres,binmeans,'ro')
    if not count:
        return centres, binmeans
    else:
        return centres,binmeans,occ

def BinnedMeanCakeByAxis(x,y,count=False):
    '''Bin along x, then average the y points in each bin. Repeat along axis 1 (phi axis) to create binned mean at each phi.
    Returns center of each bin and the mean y value in that bin.
    If binByPoints=True, n=number of points per bin.
    if binByPoints=False, n=number of bins. If n=0, numpy will estimate a good bin number ('auto' method of np.histogram).'''
    ts,means=BinnedMeanByAxis(x,y[:,0,:])
    binnd=np.zeros((means.shape[1],y.shape[1],y.shape[2]))
    for i in range(y.shape[1]): #for each phi slice, do the timebin
        ts,means=BinnedMeanByAxis(x,y[:,i,:])
        binnd[:,i,:]=means.T
    if count: #same number per bin in any slice. only count once.
        ts,means,occ=BinnedMeanByAxis(x,y[:,0,:],count=True)
        return ts,binnd,occ[0,:]
    return ts,binnd

def rebin(dat,factor,ax=0,ends='left',truncate=False):
    '''re-bin the data in dat by averaging together "factor" bins along ax. e.g. factor=2, rebin pairs of data. 
    Ends will be left hanging, by default on the left (pre-0) (ends='left'), or 'right'.
    By default end non-averaged bins are not scrapped (truncate=False), otherwise the ends will be scrapped
    together as much as possible. '''
    
    if ax != 0: #need to rebin along some other axis. Roll so that bin axis is axis 0.
        dat=np.swapaxes(dat, ax, 0)#swap axis ax into 0 position. 
    
    old_bins=dat.shape[0] #current length
    new_bins=int(old_bins/factor)
    remain=old_bins%factor
    
    ## take care of ends. 
    if ends=='left':
        good=dat[remain:] #cut off left end
        extra=dat[:remain]
    elif ends=='right':
        good=dat[:new_bins]
        extra=dat[new_bins:]
    else:
        raise ValueError('ends must be left or right')
    assert len(good)==new_bins*factor, 'length of kept bins != bin number times factor'
    assert len(extra)==remain, 'remainder wrong length'
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
    ## reshape and mean along new axis. 
        if len(dat.shape)==1:
            out=np.reshape(good,(new_bins,factor))
            out=np.nanmean(out,1)
        elif len(dat.shape)==2:
            out=np.reshape(good,(new_bins,factor,-1))
            out=np.nanmean(out,1)
        elif len(dat.shape)==3:#some t,phi,q nonsense
            out=np.reshape(good,(new_bins,factor,dat.shape[1],-1))
            out=np.nanmean(out,1)
        else:
            raise ValueError('dat has too many dimensions; cannot handle it.')

        if not truncate:
            extra=np.nanmean(extra,0) #bin those extras together

            if ends=='left':
                out=np.insert(out,0,extra,0) #put before index 0
            elif ends=='right':
                out=np.append(out,extra,0) #put on the end
    
    if ax != 0: #need to rebin along some other axis. Roll so that bin axis is axis 0.
        out=np.swapaxes(out, ax, 0)#swap axis 0 back into ax position. 
    return out


def rebin_npz(fname,nt=1,nq=1,goodq=None,ends=True,scale=True,solv_per_solute=1694, solv_peak=40.58,isocorr=True):#63.9):
    '''Open an npz file with name fname, rebin in q by nq and in t by nt. 
    if ends is false: remove 1st and last point (good for timetool mess ups)
    solv_per_solute*solv_peak gives scaling factor to convert to eu (assuming diff is already scaled by solvent peak).
    To not scale, scale to False.
    return tsr,qsr,phis,diffr, diff_2dr, isocorr'''
    dat=np.load(fname)
    qs=dat['qs']
    phis=dat['phis']
    
    if ends==False:
        ts=dat['ts'][1:-1]
        diff=np.nanmean(dat['diff'][1:-1,:,:],1) #azav that
        diff_2d=dat['diff'][1:-1,:,:]
    else:
        ts=dat['ts']
        diff=np.nanmean(dat['diff'],1) #azav that
        diff_2d=dat['diff']
        
    if scale:
        diff=diff*solv_per_solute*solv_peak
        diff_2d=diff_2d*solv_per_solute*solv_peak
    
    if goodq is None:
        goodq=np.arange(len(qs))
    tsr=rebin(ts,nt,truncate=True,ends='left')
    qsr=rebin(qs[goodq],nq,truncate=True,ends='left')
    diffr=rebin(rebin(diff[:,goodq],nt,0,truncate=True,ends='left'),nq,1,truncate=True,ends='left')
    diff_2dr=rebin(rebin(diff_2d[:,:,goodq],nt,0,truncate=True,ends='left'),nq,2,truncate=True,ends='left')
    corr_full=dat['iso_corr']
    try:
        corr_full=dat['iso_corr']
        isocorr=rebin(corr_full[:,goodq],nq,1,truncate=True,ends='left')
    except:
        isocorr=np.ones_like(diff_2dr[0,:,:])
    
    return tsr,qsr,phis,diffr, diff_2dr, isocorr
