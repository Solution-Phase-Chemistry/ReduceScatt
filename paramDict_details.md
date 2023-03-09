paramDict= {

    'binSetup'  : 'points', #'fixed' or 'unique' or 'points' or 'nbins' = (set bins; bin by unique axis values; points per bin; number of bins)
    'binSet2'   : 300, # integer points per bin for 'points' or list of bin edges for 'fixed' or integer number of bins for 'nbins'
    'binMethod' : 'ave', ## 'ave' or 'sum' ?  ### this isn't actually implemented yet. 
    
    'qnorm'     : (3,4), ## low,high or None (None=use Iscat)
    'qrange'    : (.5,4.5), ## for plots
    
    'scan_var' : None,  #variable that is being scanned
    'x_var'   : None, #binning axis, if it is not scan variable
    
    'show_filters'  : True, #show filter plots
    
    'useAzav_std'  : False, #False, 'percent', 'WAve'; if False, no azav_sqr processing, if 'percent', only use percent filter, if 'Wave' then use it to calculate weighted averages throughout
    'azav_percent_filter' : 50, # if useAzav_std='percent' or 'WAve', use azav_std as filter, reject bins where std is > this percent of bin mean value
   
    'ipm'    : 4, # select ipm to use
    'corr_filter' : True, #whether to filter based on Iscat/ipm correlation
    'corr_threshold': .03 , #threshold for correlation filter, threshold is fractional residual limit (eg. 0.03 means residuals should be <= 0.03 of the value to be kept)
    'ipm_filter' : (10000,None), #set limits for ipm intensity
    'Iscat_threshold'  : 100, #lower limit for Iscat
    'use_TT'   :   True,  #options are True, False, 'filter'  ('filter is for filter only) or 'withlxt' (time delay=lxt+encoder+ttcorrection)
    't0_corr' : None,  #false or float offset for time zero
    
    'enforce_iso' : False # enforce off shots isotropic
'energy_corr' : True, # ebeam photon energy correction (True or False)
    'NonLin_corr': None, #None, SVD, poly, or SVDbyBin
    'AdjSub'    : 50, #number of adjascent off shots to average and subtract, -1 subtracts all
    
    'aniso'  : True, # calculate anisotropy?
    'shift_n': 0, # phi offset for anisotropy in bins
    
    'xstat' : True, #calculate mean and std for x axis during binning step


'show_svd' : False #Whether to display and save an extra graph with SVD of total, S0, and S2. 
	'svd_n : 4,  #how many singular values to plot
'smooth' : None, #amount to smooth the data before SVD analysis. [q,t] where q and t are odd and represent width of bin for median filter.
'slice_plot' : None, # the q indices to mean and plot against the x variable in a 1d plot

	'overwrite' : True, # overwrite .npy files? If False then will increment file base name by 1 each time a new file is saved for the same run
    'save_mat' : False,  # save a .mat file output

    }
