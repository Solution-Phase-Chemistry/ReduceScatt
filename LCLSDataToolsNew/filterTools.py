import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import scipy
from scipy.stats import mstats
from scipy.stats.mstats import theilslopes
from sklearn.linear_model import RANSACRegressor as RSC
from sklearn.linear_model import LinearRegression as LR
import random


def FitOffset(intens1, intens2):
    '''plot intens1 vs intens2. intens1 should be something with a real zero (i.e. should be Isum). Fit a line and return intercept'''
    fit=np.polyfit(intens1,intens2,1) #linear fit between signal and normalization signal. 
    offset = fit[1] #Get offset from fit
    return offset

def slice_histogram(dat,fil,threshold,res=100,showplot=False,field='unknown',fig=None,sub='111'):
    '''For the array dat[fil], bin into a histogram with 'res' number of bins. Starting from the fullest bin,
    move outwards until a fraction of the data points are included, specified by 'threshold'.
    'field' is for labeling purposes only in error messages and plots produced when 'showplot'=True.
    Returns [leftbound, rightbound, filter]'''
    #pick out bounds around highest peak including "threshold" fraction of the counts, taking "res" # of steps
    sliceddat=dat[fil]
    sliceddat=sliceddat[~np.isnan(sliceddat)] #remove nans
    num_points=np.size(sliceddat)
    if field=='unknown':
        try:
            field=dat.name.split('/')[-1]
        except:
            pass
    #bin the data, find the mode (top of highest peak)
    counts,bins=np.histogram(sliceddat,res) 
    current_left = counts.argmax() #find the index of the fullest bin
    #print('starting from %f'%bins[current_left])
    
    #From fullest bin, move outward until "threshold" percent of data are included
    current_right=current_left+1
    number_in = float(np.size(np.where((sliceddat>bins[current_left])&(sliceddat<bins[current_right])))) #points between current bounds
    fraction_in=number_in/num_points #fraction between bounds
    left_edge= False
    right_edge = False
    while fraction_in<threshold: #until we contain the right fraction, widen the bounds. 
        #account for asymmetrical peaks: step to the direction with higher counts
        try:
            leftnum=counts[current_left-1]
        except IndexError:
            #we have reached the left edge. Can't go further, step to the right
            if not left_edge:
                print('hit left edge on %s' % (field))
            left_edge=True;
            showplot=True
        try:
            rightnum=counts[current_right+1]
        except IndexError:
            #we have reached the right edge. Can't go further, step to the left regardless
            if not right_edge:
                print('hit right edge on %s' % (field))
            right_edge=True;
            showplot=True
            
        #if we've hit an edge, there is no choice. Step the other way
        if left_edge:
            current_right=current_right+1
        elif right_edge:
            current_left=current_left-1
        else:
            #if we're not on the edge, pick higher direction
            if leftnum > rightnum:
                current_left=current_left-1
            else:
                current_right=current_right+1
        try:
            number_in = np.size(np.where((sliceddat>bins[current_left])&(sliceddat<bins[current_right])))
        except IndexError: #if the mode is zero and we hit the other edge, hack a solution (do better...)
            current_left = current_left + 1
            current_right =current_right - 1
            fraction_in=number_in/num_points
            print('edges reached; settle for fraction %f' %(fraction_in))
            break
        fraction_in=float(number_in)/num_points
        #print(fraction_in)
    
    if showplot:
        if fig is None:
            plt.figure()
        else:
            plt.figure(fig)
            plt.subplot(sub)
        plt.hist([sliceddat[np.where((sliceddat>bins[current_left])&(sliceddat<bins[current_right]))],
                  sliceddat[np.where((sliceddat<bins[current_left])|(sliceddat>bins[current_right]))]],
                 res,color=['r','k'],histtype='barstacked')
        plt.title(field+ ' histogram slice')
    filter_out=(dat>bins[current_left])&(dat<bins[current_right])#true/false of all points between bounds
    filter_out=filter_out & fil #only apply it to points within the input filter
    return bins[current_left], bins[current_right],fraction_in,filter_out




def slope_filter(xin, yin,fil, slope, intercept=0):
    f_slope=np.full_like(yin,False)
    slope_in=np.full_like(yin[fil],False).astype(bool)
    slope_in[yin[fil] >= (xin[fil]*slope+intercept)]=True
    
    print('fraction kept ', yin[fil][slope_in.astype(bool)].shape[0]/yin[fil].shape[0])
    
    f_slope[fil]=slope_in
    
    
    return f_slope.astype(bool)
        




def correlation_filter_RANSAC(ipm1, Isum1,thresh,subset=None,intercept=True):
    '''use ransac algorithm to find line of best fit to correlation data.
        to do:  also add parallelized version.
        
        thresh: threshold for residual such that if residuals are larger points are rejected
        subset: use a subset of the data to fit line;
        intercept: boolean, if true fit eqn is mx+b, if false fit eqn is mx
        '''

    ## correlation filter of ipm vs Isum
    
    #RANSAC fit to line
    RSCthresh=10 #inlier/outlier threshold
    trialN=20 #number of RANSAC trials to preform
    trialsRSC=np.arange(trialN)
    outRSC=np.full((trialN,2),np.nan)
    for i in trialsRSC:
        
        if subset is not None:
            temp_indx=np.arange(ipm1.shape[0])
            random.shuffle(temp_indx)
            indx=temp_indx[:subset] #choose random subset of indices
        else:
            indx=np.arange(ipm1.shape[0])
            
        #fit_intercept=False: y=mx
        reg=RSC(estimator=LR(fit_intercept=intercept), 
                residual_threshold=RSCthresh,max_trials=100,is_data_valid=None).fit(ipm1[indx],Isum1[indx])

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
    print('slope std',np.nanstd(outRSC[:,0]))
    print('intercept std',np.nanstd(outRSC[:,1]))
    print('fraction of data kept %e' %(Isum1[in_mask].shape[0]/Isum1.shape[0]))
    

    return in_mask, line_y




###########################
#Work in progress
###########################





def correlation_filter_Theil(xin,yin,fil,xray_on,threshold,showplot=False):
    '''Fits a line to (xin[fil],yin[fil]). Then, of all points with xray_on, pick points closest to this line until 
    a fraction of the points are included, specified by 'threshold'. Returns [correlation_filter, correlation_filter&fil]. '''
    x=xin[fil]
    y=yin[fil]
    #correlation filter. Only accept points where x and y are correlated, keeping "threshold" fraction of the points.
#    if showplot:
#        fig=plt.figure('scatter')
#        plt.plot(x,y,'b.')

    #find slope using Theil-Sen (to ignore outliers)
    
    try:
        #print(np.size(y))
        #print(np.size(x))
        bits=int(np.floor(np.linspace(0,len(x),1000)))#3000 spaced out points from the array to fit on
        print(bits)
        m,b,u,l=theilslopes(y[bits],x[bits])#fit on a subset of the data because this function scales with factorial 
        #and memory is insufficient
    except (IndexError, TypeError): #there are less than 6000 points so just use all of them
        m,b,u,l=theilslopes(y,x)
    fitval=np.polyval([m,b],xin)
    #keep points within threshold, choose threshold to keep % of points
    distance=np.array(yin-fitval, dtype=[('diff', float)]) #distance of each point from line
    (lower, upper, corr_filter)=slice_histogram(distance,'diff',xray_on,threshold) #keep those close to mean, which have the x-ray on
    
    if showplot:
        plt.title('distribution around line of best fit') 
        plt.figure('scatter')
        plt.plot(xin[corr_filter][bits], yin[corr_filter][bits],'m.')
        plt.plot(xin[fil][bits], fitval[fil][bits],'g.')
        closefig()
    combined_filter=fil&corr_filter
    return corr_filter, combined_filter

