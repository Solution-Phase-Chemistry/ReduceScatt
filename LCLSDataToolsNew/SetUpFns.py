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

from sklearn.linear_model import RANSACRegressor as RSC
from sklearn.linear_model import LinearRegression as LR

''' Set up functions and filters '''

def LoadH5(fname,outpath,varDict,paramDict,outDict):
    ''' load values designated in varDict from H5 file with name fname
    save data to outDict'''
    overwrite=paramDict['overwrite']
    
    basename=fname.split('/')[-1].split('.')[0]
    basename1=basename
    if not overwrite and os.path.isfile(outpath+'npy/'+basename+'_out.npy'):
        nn=0
        while os.path.isfile(outpath+'npy/'+basename1+'_out.npy'):
            nn+=1
            basename1=basename+'_%02i' %nn
    print('basename is ', basename1)
    outDict['h5name']=basename1
    h5Dict={}
    with h5py.File(fname,'r') as d:
        for key in varDict:
            try:
                h5Dict[key]=d[varDict[key]][:]
            except:
                print('key not found '+ key)
        if paramDict['scan_var'] is None:
            try:
                if np.unique(h5Dict['scan_vec']).size > 1:
                    for scanvar in d['scan']:
                        paramDict['scan_var']=scanvar
                        break
                else:
                    paramDict['scan_var']='newdelay'

            except:
                h5Dict['scan_vec']=h5Dict['encoder']
                paramDict['scan_var']='newdelay'
            
    print('scan variable is ', paramDict['scan_var'])
    outDict['h5Dict']=h5Dict
    print('finished loading h5')
    
    
def AzavStdFilter(paramDict,outDict):
    '''calculate Azav_Std from azav and azav_sqr fields of h5.  
    use parameter azav_percent_filter to reject azav bins where  azav_Std > perLim % is rejected
    '''
    
    azav_temp=outDict['h5Dict']['azav']
    azav_sqr=outDict['h5Dict']['azav_sqr']
    perFilt=paramDict['azav_percent_filter']
    
    azav_std=np.sqrt(np.abs(azav_sqr-(azav_temp)**2))
    
    # ## check if this is the formula we should be using
    # pix_per_bin=outDict['h5view']['pix_per_azav']
    # azav_std=np.sqrt(np.abs(azav_sqr/pix_per_bin-(azav_temp)**2))
    
    
    if perFilt is not None:
        ## remove noisy bins;  #error > perFilt % is rejected
        azav_per=np.nan_to_num(azav_std/azav_temp*100) #percent error
        azav_temp[azav_per>perFilt]=np.nan 
        azav_std[azav_per>perFilt]=np.nan
    
    outDict['h5Dict']['azav']=azav_temp
    outDict['h5Dict']['azav_std']=azav_std
    
    print('AzavStdFilt - done!')

    
    
    
def MaskAzav(paramDict,outDict,listBinInd=None):
    ''' Mask specific azav bins and first/last two q bins for each phi 
    listBinInd = n x 2 array of phi,q indices for bin(s) you want to delete
    ex. [0,269]''' 
    print('masking azav bins')
    
    if paramDict['useAzav_std']=='WAve':
        keyList=['azav','azav_std']
    else:
        keyList=['azav']
        
    for key in keyList:
        azav_temp=outDict['h5Dict'][key]
        if listBinInd is not None:
            for ii in range(listBinInd.shape[0]):
                azav_temp[:,listBinInd[ii,0],listBinInd[ii,1]]=np.nan

        ind=np.nonzero(~np.isnan(azav_temp))
        iPhi=ind[1]
        iQ=ind[2]

        #for each phi bin replace first and last not-nan q bin with nan
        for i in range(azav_temp.shape[1]):
            ind2=np.nonzero(iPhi==i) #isolate this phi bin
            qind=iQ[ind2] #already sorted
            azav_temp[:,i,qind[[0,1,-2,-1]]]=np.nan
        
        outDict['h5Dict'][key]=azav_temp

    
    print('masking azav bins - done!')

    
    
    
def setupFilters(paramDict,outDict):
    ''' calculate Iscat and set up laser on/off etc filters'''
    
    ## calculate Iscat aka Isum (but actually average?) check this
    azav_temp=outDict['h5Dict']['azav']
    Iscat=np.nanmean(azav_temp,(1,2)) #mean along 2 axes
    outDict['Iscat']=Iscat
    outDict['numshots']=azav_temp.shape[0]
    print('calculated Iscat') 
    
    
    ### set up filters
    if 'filters' not in outDict.keys():
        outDict['filters']={}
    outDict['filters']['f_xon']=outDict['h5Dict']['xray_status']==1
    outDict['filters']['f_lon']=outDict['h5Dict']['laser_status']==1
    outDict['filters']['f_xoff']=outDict['h5Dict']['xray_status']==0
    outDict['filters']['f_loff']=outDict['h5Dict']['laser_status']==0 
    ## for later:
    outDict['filters']['f_lgood']=outDict['filters']['f_lon'] 
    outDict['filters']['f_good']= outDict['filters']['f_xon']
    print('setupFilters - done!')
    
    
    

def IscatFilters(paramDict,outDict):
    '''
    if corr_filter then use ipm thresholds and Iscat/ipm correlation fit to filter
    histogram filter of Iscat 80%'''
    
    Iscat=outDict['Iscat']
    f_xon=outDict['filters']['f_xon']
    Iscat_thresh=paramDict['Iscat_threshold']
    ipm_thresh=paramDict['ipm_filter']
    
    
        ##create filter on xray intensity keeping 80% of shots
    l,r,frac,f_Iscat=slice_histogram(Iscat,f_xon&(Iscat>Iscat_thresh),0.80, 
                                      showplot=paramDict['show_filters'], fig='red', field='Iscat',sub=221)
    outDict['filters']['f_Iscat']=f_Iscat
    outDict['filters']['f_good']=f_Iscat&f_xon ##formerly known as f_intens
    
    
    ipmkey='ipm'+str(paramDict['ipm'])
    ipmi=outDict['h5Dict'][ipmkey]
    
    ### ipm_thresholds
    if paramDict['ipm_filter'][0] != None:
        f_Ipm=(ipmi>ipm_thresh[0])
    if paramDict['ipm_filter'][1] != None:
        f_Ipm=f_Ipm & (ipmi<ipm_thresh[1])  
    outDict['filters']['f_Ipm']=f_Ipm
    outDict['filters']['f_good']=outDict['filters']['f_good'] & f_Ipm
    
    if paramDict['corr_filter']:
        print('making correlation filter')
        
        thresh=paramDict['corr_threshold']
        
        ##set ipm thresholds
        # ipmkey='ipm'+str(paramDict['ipm'])
        # ipmi=outDict['h5Dict'][ipmkey]
        ipmfilt=np.full_like(ipmi,1)
        if paramDict['ipm_filter'][0] != None:
            ipmfilt=np.logical_and((ipmfilt),(ipmi>ipm_thresh[0]))
        if paramDict['ipm_filter'][1] != None:
            ipmfilt=np.logical_and((ipmfilt),(ipmi<ipm_thresh[1]))
        
    
        ## correlation filter of ipm vs Isum
        # thresh=30
        
        #RANSAC fit to line
        nanfilt=~np.isnan(Iscat)&~np.isnan(ipmi)&(ipmfilt)&(Iscat>Iscat_thresh)
        ipm1=np.expand_dims(ipmi[nanfilt],axis=1)
        Isum1=Iscat[nanfilt]

        RSCthresh=10 #inlier/outlier threshold
        trialN=100 #number of RANSAC trials to preform
        trialsRSC=np.arange(trialN)
        outRSC=np.full((trialN,2),np.nan)
        for i in trialsRSC:
            #fit_intercept=False: y=mx
            reg=RSC(estimator=LR(fit_intercept=True), 
                    residual_threshold=RSCthresh,max_trials=10000,is_data_valid=None).fit(ipm1,Isum1)

            outRSC[i,0]=reg.estimator_.coef_[0] #slope
            outRSC[i,1]=reg.estimator_.intercept_ #intercept 


        #find average slope 
        mm=np.nanmean(outRSC[:,0])
        bb=np.nanmean(outRSC[:,1]) 
        #find inliers
        residF=np.abs((mm*ipm1.squeeze()+bb)-Isum1.squeeze())/Isum1.squeeze()
        in_mask=residF<=thresh #inliers
        out_mask=residF>thresh #outliers
        line_y=ipm1[in_mask]*mm+bb
        print('correlation equation = %e x +%e' %(mm,bb))
        print('fraction of data kept %e' %(Isum1[in_mask].shape[0]/Isum1.shape[0]))
        f_corr=np.zeros(Iscat.shape).astype(bool)
        f_corr[nanfilt]=in_mask
        outDict['filters']['f_corr']=f_corr
        outDict['filters']['f_good']=outDict['filters']['f_good'] & f_corr
                    
        ## plot
    if paramDict['show_filters']: 
        plt.figure('red')
        plt.subplot(2,2,2)
        if paramDict['corr_filter']:
            plt.hist2d(ipmi[nanfilt].squeeze(),Iscat[nanfilt].squeeze(),100,cmap='Greys',
                       norm=mpl.colors.SymLogNorm(linthresh=1, linscale=1))
            plt.scatter(ipm1[in_mask],Isum1[in_mask],marker='.',color='blue',alpha=.1)
            plt.plot(ipm1[in_mask],line_y,color='r')
            plt.xlim(left=0)
        else:
            nanfilt=~np.isnan(Iscat)&~np.isnan(ipmi)
            plt.hist2d(ipmi[nanfilt].squeeze(),Iscat[nanfilt].squeeze(),100,cmap='Greys',
                       norm=mpl.colors.SymLogNorm(linthresh=1, linscale=1))
          
        if paramDict['ipm_filter'][0] != None:
            plt.axvline(ipm_thresh[0],ls='--',color='r')
        if paramDict['ipm_filter'][1] != None:
            plt.axvline(ipm_thresh[1],ls='--',color='r')
        # plt.ticklabel_format(scilimits=(-3,3))
        plt.xlabel(ipmkey)
        plt.ylabel('Iscat')
        plt.title('hist of shots (log colorbar)')

            
            
            
            
def laserFilter(paramDict, outDict):
    ''' filter on laser intensity'''
    laserP= outDict['h5Dict']['laser_diode'][:,0]
    f_lon=outDict['filters']['f_lgood']
    
    #filter on the laser intensity
    
    # filterP = laserP>(np.nanmean(laserP)-np.nanstd(laserP)*3) 
    # l,r,frac,f_laser=slice_histogram(laserP,(f_lon&filterP),0.99)

    l,r,frac,f_laser=slice_histogram(laserP,(f_lon),0.99)
    
    outDict['filters']['f_laser']=f_laser
    outDict['filters']['f_lgood']=f_lon&f_laser
    print('filter on laser intensity -Done!')
    
    
    
    
    
def eBeamFilter(paramDict,outDict):
    '''filter on x-ray photon energy/ebeam energy'''
    ebeam=outDict['h5Dict']['ebeam_hv']
    f_good=outDict['filters']['f_good']
    
    
    ## if fixed limits:
    # lower=17000
    # upper=17200
    # f_energ=(d['ebeam/photon_energy'][:]>lower)&(d['ebeam/photon_energy'][:]<upper) #necessary for the energy correction to be well-fit
    # frac=float(np.sum(f_energ))/float(len(f_energ))
    
    ##use slice histogram to keep 99% of shots
    l,r,frac,f_energ=slice_histogram(ebeam,f_good,
                                      0.99,showplot=False)
    
    outDict['filters']['f_eBeam']=f_energ
    outDict['filters']['f_good']=f_good&f_energ
    print('ebeam filter: fraction_kept ',frac,' lower ', l,' upper ',r)
    
    
    
    


def TTfilter(paramDict,outDict):

    ttfwhm=outDict['h5Dict']['ttFWHM']
    ttpos=outDict['h5Dict']['ttFLTPOS']
    ttamp=outDict['h5Dict']['ttAMPL']
    showfilt=paramDict['show_filters']
    f_lon=outDict['filters']['f_lgood']
    f_good=outDict['filters']['f_good']

    ## filter on TT amplitude
    
    f_ttamp=(ttamp>0.02) #add this in if ttool seems suspect
    
    

    ##filter based on TT fwhm
    l,r,frac,f_ttfwhm=slice_histogram(ttfwhm,
                                      f_lon&f_good&(ttfwhm>10)&(ttfwhm<300),
                                      0.99,showplot=showfilt,fig='red',field='TTfwhm',sub=223)
    print('TTFWHM: fraction_kept ',frac,' lower ', l,' upper ',r)

    ## filter based on TT position
    l,r,frac,f_ttpos=slice_histogram(ttpos,
                                      (f_good&f_ttamp&(ttpos>10)),
                                      0.99,showplot=showfilt,field='TTpos',fig='red',sub=224)
    print('TTPOS: fraction_kept ',frac,' lower ', l,' upper ',r)

    ## save
    outDict['filters']['f_allTT']=f_ttpos&f_ttfwhm&f_ttamp
    outDict['filters']['f_lgood']=f_lon&f_ttpos&f_ttfwhm&f_ttamp
    outDict['filters']['f_good']=(outDict['filters']['f_good']&outDict['filters']['f_loff'])|(outDict['filters']['f_good']&outDict['filters']['f_lgood']) #want f_intens to cut on timetool as well



def saveReduction(outDir,paramDict,outDict):
    basename=outDict['h5name']
    
    plt.figure('red')
    plt.suptitle(basename)
    figdir = outDir + 'figures/'
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(figdir+basename+'_reduction.png')   
    print('saved reduction.png')
    
    
    

def EnforceIso(paramDict,outDict):
    ''' enforce isotropic off shots'''
    f_loff=outDict['filters']['f_loff']
    f_intens=outDict['filters']['f_good']
    azav=outDict['h5Dict']['azav']
    Isum=outDict['Iscat']
    corr=np.array([0])
    if ((np.sum(corr)/corr.size)<0.8) or (np.nanmax(corr)>4):
        random_sample=np.random.randint(sum(f_loff&f_intens),size=400) #500 random off shots
        #print(random_sample)
        norm_offs=divide_anyshape(azav[f_loff&f_intens,:,:][random_sample],Isum[f_loff&f_intens][random_sample])
        average_off=np.nanmean(norm_offs,0)#only use 500 to speed up calc
        #print(average_off.shape)
        average_slice=np.nanmean(average_off,0)
        #print(average_slice.shape)
        corr=divide_anyshape(average_off,average_slice)
        print('trying correction')

    print('corr found.')

    #print(corr.shape)
    plot_cake(corr,fig='iso',sub=111)
    plt.suptitle('isotropic off shot correction')
    #corr_cspad=d['cspad']['azav']/corr
    
    outDict['iso_corr']=corr
