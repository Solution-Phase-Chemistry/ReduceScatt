# paramDict options and details


###### 'binSetup': type of binning of scanvariable
- 'fixed' specify array of bin edges 
- 'unique' bin by unique axis values 
- 'points' specify number of points per bin 
- 'nbins'  specify number of bins
###### 'binSet2' :  more binning parameters
- integer points per bin for 'binSetup': 'points'
- list/array of bin edges (float, in seconds) for 'binSetup': 'fixed'
- integer number of bins for 'binSetup': 'nbins' 
###### 'binMethod' : how to bin
- 'ave' calculates average value of points in bin
- 'sum' calculates sum of points in bin (this isn't implemented yet)


###### 'qnorm' : qrange for normalization
- (float,float) = low, high limits of q range used for normalization
- None = use Iscat for normalization
###### 'qrange' : qrange for plots
- (float,float) q range used for plots and SVD
###### 'scan_var' : variable that is being scanned
- None : most of the time this can be left as None and scan variable will automatically be read from h5 file.  If it is not being read then specifiy it here (eg. 'lensh', or 'newdelay' for a delay scan)
###### 'x_var' : binning axis, if it is not same as scan variable
- None : most of the time leave this as None
###### 'show_filters' : show filter plots
- True show plots
- False don't show plots
###### 'useAzav_std' :  if azav_sqr is saved in h5 file, this parameter dictates how it is used
- False if no azav_sqr processing
- 'percent' if only use azav_sqr as a percent filter (reject points for which variance is greater than some percent of azav)
- 'Wave' use azav_sqr to calculate variance, use variance to calculate weighted averages throughout reduction
###### 'azav_percent_filter' : if 'useAzav_std' is not False, reject bins where std is > this percent of bin mean value
- float percentage 


###### 'ipm' : select ipm to use as upstream I0
- integer (for XCS this is either 4 or 5)
###### 'corr_filter' : whether to filter based on Iscat vs I0 correlation
- True
- False
###### 'corr_threshold' : threshold for correlation filter, threshold is fractional residual limit (eg. 0.03 means residuals should be <= 0.03 of the value to be kept)
- float
###### 'ipm_filter' :  set low and high limits for ipm readout
- (float,float), (float,None), (None,float), (None,None)
###### 'Iscat_threshold' :  lower limit for Iscat value
- float
###### 'use_TT' : whther and how to use time tool information
- True use time tool to calculate delay axis and as a filter
- False do not use time tool at all
- 'filter'  only use time tool as a filter
- 'withlxt'  use time tool, encoder, and lxt to calculate delay axis (time delay=lxt+encoder+ttcorrection), use timetool as filter
###### 't0_corr' : constant offset for time zero (eg. if delay axis reads 1 ps at time zero (difference signal onset), set this to 1 ps)
- None
- float (seconds)


###### 'enforce_iso' : enforce isotropic off shots
- True
- False
###### 'energy_corr' : ebeam photon energy correction
- True
- False
###### 'NonLin_corr' : detector nonlinearity (with respect to intensity) correction
- None
- 'SVD'  uses SVD correction on azimuthally averaged signal
- 'SVDbyBin' uses SVD correction for each phi bin
###### 'AdjSub' :  number of adjascent off shots to average and subtract
- integer
- -1 to subtract all off shots
###### 'BackSub' : Subtract t<0 signal from all data
- None do not do back subtraction
- 'SVD' take the SVD of the t<0 data and subtract the major component
- 'ave' subtract the average of the t<0 data
###### 'earlytrange': time range to use fr t<0 data in 'BackSub'
- (float,float) in seconds


###### 'aniso' : calculate anisotropy?
- True
- False
###### 'shift_n' : phi offset for anisotrpoy calculation in bins
- integer 
###### 'xstat' : calculate mean and standard deviation for x-axis during binning step?
- True
- False
###### 'show_svd' : Calculate SVD of scattering signal (and S0, S2 if applicable), display and save figure
- True
- False
###### 'svd_n' : if 'show_svd', how many singular values to plot
- integer
###### 'smooth' : if 'show_svd' amount to smooth data before SVD analysis
- None
- (integer,integer)  = (bin_q, bin_t) where both are odd and represent width of bin (in points) for median filter 
###### 'slice_plot' : determines what is plotted in the results overview plot
- None:  plot every other time slice, DiffSig vs Q
- [float,float]: 1D plot of data vs time averaged over q range defined by [qlow, qhigh]



###### 'overwrite' :  overwrite .npy files? 
- True
- False then will increment file base name by 1 each time a new file is saved for the same run
###### 'save_mat' : save a .mat file output
- True
- False

###### 'save_h5' : save a .h5 file output
- True
- False

## paramDict defaults

    paramDict= {
            'binSetup'  : 'points',
            'binSet2'   : 300,
            'binMethod' : 'ave', 
            'qnorm'     : (3,4), 
            'qrange'    : (0.5,4),
            'show_filters'  : True, #show filter plots
            'useAzav_std'  : False, 
            'azav_percent_filter' : None, 
                        
            'ipm'    : 5, # select ipm to use for I0
            'corr_filter' : True,
            'corr_threshold': .03 , 
            'ipm_filter' : (1000,None), #set limits for ipm intensity
            'Iscat_threshold'  : 10, #lower limit for Iscat

            'use_TT'   :  True,  
            'scan_var' : None,
            'x_var'   : None,

            't0_corr' : None,  #false or float offset for time zero
            'enforce_iso': False, #enforce isotropic off shots?
            'energy_corr' : False, # ebeam photon energy correction
            'NonLin_corr': None, #None, SVD, poly, or SVDbyBin

            'AdjSub'    : 50, #number of adjascent off shots to average and subtract, -1 subtracts all
            'aniso'  : False,
            'shift_n' :0,
            'xstat' : True,

            'BackSub': None,
            'earlytrange':(-0.5e-12,0e-12),

            'showSVD': False, #do svd
            'SVD_n': 4, # number of svd components
            'slice_plot':None, #how to slice up plots displayed at end
            'smooth':None, #smoothing before svd?

            'overwrite' : True, # overwrite files? 
            'save_mat'  : False,
            'save_h5' : True,

        }



