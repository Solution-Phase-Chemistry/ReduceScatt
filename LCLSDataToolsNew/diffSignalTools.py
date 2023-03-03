import numpy as np
from matplotlib import pyplot as plt
import warnings

def highq_normalization_factor(dat, qs, qlow, qhigh):
    '''make a normalization factor out of the average of the higher-q data. 
    dat=azav
    qs=azav.q
    qlow,qhigh=limits of ROI for normalization'''
    qs=qs[:]
    az_av=np.nanmean(dat,1) #single azimuthal averages
    q_ind=(qs>qlow)&(qs<qhigh)
    norm_factor=np.nanmean(az_av[:,q_ind],1) #mean of points between q limits for each shot
    return norm_factor

def norm_factor_phis(dat, qs, qlow, qhigh):
    '''make a normalization factor out of the average of the higher-q data, for each phi bin. 
    dat=azav
    qs=azav.q
    qlow,qhigh=limits of ROI for normalization'''
   
    qs=qs[:]
    az_av=dat[:,:,:] #single azimuthal averages
    print(qs.shape, az_av.shape)
    q_ind=(qs>qlow)&(qs<qhigh)
    norm_factor=np.nanmean(az_av[:,:,q_ind],2) #mean of points between q limits for each shot
    return norm_factor

def divide_anyshape(lg,sm):
    '''divide array lg of shape (N,), (N,M,) or (N,M,L) by array sm of shape (N,)'''
    try:
        out=lg/sm
    except ValueError:
        try:
            out=lg/sm[:,np.newaxis]
        except ValueError:
            out=lg/sm[:,np.newaxis,np.newaxis]
    return out

def DifferenceSignal(dat,f_good,f_lon,f_loff,n,offset=0,totaloff=1):
    '''dat is normalized signal to be differenced. 
    nor is normalization factor (e.g. ipm2)
    f_good, f_on, f_off: filters for good x-ray shots, laser on and off shots
    returns difference signal for all good, laser on shots; filled with nan to be the same shape as other data fields 
    (for later binning)
    n is the number of off shots to average to subtract from ons. If n<0, average all offs. 
    offset is calculated externally to account for nor=0 not corresponding to signal=0.'''
    n=int(n)
    if n%2==0: #only works for odd n; if even, just add one
        n+=1
    norm_dat=np.squeeze(dat[:])
    # nor=np.squeeze(nor[:])
    diff=np.zeros(norm_dat.shape)*np.nan #fill with nan. Points with a difference signal will get numbers. 
    norm_off=norm_dat[f_good&f_loff]
    
    if n<0:
#         print('this one')
        #average all offs and subtract from all ons. 
        bkg=np.nanmean(norm_off,0)#keep shape of dat, mean only along 0 axis
        diffsig=(norm_dat[f_good&f_lon]-bkg)/totaloff
        diff[f_good&f_lon]=diffsig
 
    else:
        #running average of n offs and subtract from adjacent ons.
        drops=np.where(f_loff&f_good)[0] #list of indices of laser-off shots. 
        for i,shot in enumerate(drops):
            #sys.stdout.write('\r'+str(i))
            #sys.stdout.flush()
            if i<n/2: #on the ends, things will get weird. 
                #weird stuff
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
                    bkg=np.nanmean(norm_off[0:n],0)
                #print(bkg.shape)
                diff[0:drops[i]]=(norm_dat[0:drops[i]]-bkg)/totaloff
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
                        bkg=np.nanmean(norm_off[int(i-n/2):int(i+n/2)],0) #e.g. n=5, i=5 average norm_off[2:7]
                    diff[drops[i]:drops[i+1]]=(norm_dat[drops[i]:drops[i+1]]-bkg)/totaloff
                except IndexError: #got to the end of the list
                    #print('end of line')
                    #average last n and subtract from all
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
                        bkg=np.nanmean(norm_off[-n:],0)
                    diff[drops[i]:]=(norm_dat[drops[i]:]-bkg)/totaloff     
    return diff





def DifferenceError(dat,derr,f_good,f_lon,f_loff,n,offset=0,totaloff=1):
    '''dat is normalized signal to be differenced. 
    nor is normalization factor (e.g. ipm2)
    derr=std for dat 
    f_good, f_on, f_off: filters for good x-ray shots, laser on and off shots
    returns error propagated standard deviation for difference signal for all good, laser on shots; filled with nan to be the same shape as other data fields 
    (for later binning)
    n is the number of off shots to average to subtract from ons. If n<0, average all offs. 
    offset is calculated externally to account for nor=0 not corresponding to signal=0.
    totaloff=scaling factor,  often totaloff=np.nanmax(np.nanmean(cake,0))=peak of laser off scattering signal. 
    
    '''
    n=int(n)
    if n%2==0: #only works for odd n; if even, just add one
        n+=1
    dat=np.squeeze(dat[:])
    
    norm_derr=np.squeeze(derr[:])
    diff_err=np.zeros(dat.shape)*np.nan #fill with nan. Points with a difference signal will get numbers. 

    
    norm_off=norm_derr[f_good&f_loff]
    if n<0:
        #Propagate error for: average all offs and subtract from all ons. 
        bkg_err=np.sqrt(np.nansum(norm_off**2,0))/norm_off.shape[0]
        err_temp=np.sqrt(norm_derr[f_good&f_lon]**2+bkg_err**2)/totaloff
        diff_err[f_good&f_lon]=err_temp
 
    else:
        #running average of n offs and subtract from adjacent ons.
        drops=np.where(f_loff&f_good)[0] #list of indices of laser-off shots. 
        for i,shot in enumerate(drops):
            #sys.stdout.write('\r'+str(i))
            #sys.stdout.flush()
            if i<n/2: #on the ends, things will get weird. 
                #weird stuff
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
                    
                    bkg_err=np.sqrt(np.nansum(norm_off[0:n]**2,0))/norm_off[0:n].shape[0]
                
                diff_err[0:drops[i]]=np.sqrt(norm_derr[0:drops[i]]**2+bkg_err**2)/totaloff
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
                        bkg_err=np.sqrt(np.nansum(norm_off[int(i-n/2):int(i+n/2)]**2,0))/norm_off[int(i-n/2):int(i+n/2)].shape[0] #e.g. n=5, i=5 average norm_off[2:7]
                    diff_err[drops[i]:drops[i+1]]=np.sqrt(norm_derr[drops[i]:drops[i+1]]**2+bkg_err**2)/totaloff
                except IndexError: #got to the end of the list
                    #print('end of line')
                    #average last n and subtract from all
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning) #get rid of obnoxious warning when slice is all nan
                        bkg_err=np.sqrt(np.nansum(norm_off[-n:]**2,0))/norm_off[-n:].shape[0]                          
                    diff_err[drops[i]:]=np.sqrt(norm_derr[drops[i]:]**2+bkg_err**2)/totaloff     
    return diff_err
