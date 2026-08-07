# Scattering Code Information


# outDict Reference:
## list of all keys in outDict


    h5name
    numshots
    h5Dict  #don't save in npy
        #load based on variable dictionary
    filters # not saved in .mat
        f_lon
        f_loff
        f_xon
        f_xoff
        f_good
        f_lgood
        f_allTT
        f_Iscat
	f_Ipm
        f_laser
        f_corr
        f_eBeam
    Iscat    #don't save in npy
    xs #don't save in npy
    normal_factor
    loff_cake
    iso_corr
    x_Data #don't save in npy
    diff_Data  #don't save in npy
    diff_bin = time binned data
    diff_std =  std for each time bin
    diff_azav_std = propagated std for each q/phi bin
    xbin_occupancy
    xcenter
    xmean
    xstd
    S0
    S0_err
    S2
    S2_err
    
    qs
    phis
    numshots_used
    paramDict #not saved in .mat

# variable dictionary reference
	varDict = {
		'ipm4'          : 'ipm4/sum',
		'ipm5'          : 'ipm5/sum',
		'xray_status'   : 'lightStatus/xray',
		'laser_status'  : 'lightStatus/laser',
		'ebeam_hv'      : 'ebeam/photon_energy',
		'scan_vec'      : 'scan/var0',
		'laser_diode'   : '/diodeGon/channels/',

		## tt variables
		'ttCorr'        : 'tt/ttCorr',
		'ttAMPL'        : 'tt/AMPL',
		'ttFWHM'        : 'tt/FLTPOSFWHM',
		'ttFLTPOS'      : 'tt/FLTPOS',
		# 'ttFLTPOS_PS'   : 'tt/FLTPOS_PS', ## in picoseconds
		# 'ttREFAMPL'     : 'tt/REFAMPL',
		'encoder'       : 'enc/lasDelay',
		'lxt'           : 'epics/lxt' , 
		# 'lxt_ttc'       : 'epics/lxt_ttc',

		## scattering variables
		'azav'      : 'epix10k2M/azav_azav',
		'qs'        : 'UserDataCfg/epix10k2M/azav__azav_q',
		'phis'      :  'UserDataCfg/epix10k2M/azav__azav_phiVec',

		## scattering detector error
		'azav_sqr'     :  'epix10k2M/azav_azav_square',
		'pix_per_azav'  :  'UserDataCfg/epix10k2M/azav__azav_norm',

		## second scattering detector
		'azav'      : 'epix10k135/azav_azav',

		## spectroscopy variables
		'epix_roi0'        : 'epix_1/ROI_0_area',
		'epix_roi0_sum'    : 'epix_1/ROI_0_sum',
		'droplet_x'        : 'epix_1/ragged_droplet_photon_j', #nondispersive axis
		'droplet_y'        :  'epix_1/ragged_droplet_photon_i', #energy dispersive
		'epix_roi0_limits' : 'UserDataCfg/epix_1/ROI_0__ROI_0_ROI',

		}






