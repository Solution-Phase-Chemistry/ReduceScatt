# Scattering Code Information
# paramDict Reference:
paramDict= {
        'binSetup'  : 'points', #'fixed' OR 'unique' OR 'points' OR 'nbins' = (set bins; bin by unique axis values; points per bin; number of bins)
        'binSet2'   : 300, # integer points per bin for 'points' or list of bin edges for 'fixed' or integer number of bins for 'nbins'
        'binMethod' : 'ave', ## 'ave' or 'sum' ? 
        
        'qnorm'     : (3,4), ## low,high or None (None=use Iscat)
        'qrange'    : (.5,4.5), ## for plots
        
        'scan_var' : None,  #variable that is being scanned
        'x_var'   : None, #binning axis, if it is not scan variable
        
        'show_filters'  : True, #show filter plots
        
        'useAzav_std'  : False, #False, 'percent', 'WAve'; if False, no azav_sqr processing, if 'percent', only use percent filter, if 'Wave' then use it to calculate weighted averages throughout
        'azav_percent_filter' : 50, # if useAzav_std='percent' or 'WAve', use azav_std as filter, reject bins where std is > this percent of bin mean value
       
        'ipm'    : 4, # select ipm to use
        'corr_filter' : True, #whether to filter based on Iscat/ipm correlation
        'ipm_filter' : (10,None), # if corr_filter, can set limits for ipm intensity
        'Iscat_threshold'  : 50, #reject values less than this for Iscat (aka Isum)
        'use_TT'   :   True,  #options are True, False, and 'filter'  ('filter is for filter only) 
        't0_corr' : None,  #false or float offset for time zero
        
        'enforce_iso' : False # enforce off shots isotropic
        'NonLin_corr': None, #None, SVD, poly, or SVDbyBin
        'AdjSub'    : 50, #number of adjascent off shots to average and subtract, -1 subtracts all
        
        'aniso'  : True, # calculate anisotropy?
        'shift_n': 0, # phi offset for anisotropy in bins
        
        'xstat' : True, #calculate mean and std for x axis during binning step
 

	   show_svd=False : Whether to display and save an extra graph with SVD of total, S0, and S2. 
    svd_n=4 : how many singular values to plot

    smooth=None : amount to smooth the data before SVD analysis. [q,t] where q and t are odd and represent width of bin for median filter.

    
    slice_plot=None : the q indices to mean and plot against the x variable in a 1d plot

    
        }

# outDict Reference:
## list of all keys in outDict


    h5name
    numshots
    h5Dict  #don't save in npy
        #load based on variable dictionary
    filters
        f_lon
        f_loff
        f_xon
        f_xoff
        f_good
        f_lgood
        f_allTT
        f_Iscat
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
    paramDict

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




# Where is my function?
## anisotropyToolsAll.py
		def theil_sen_stats(y,x) \
		def SOS2(dat, phis, fil=None, shift_n=0,deg=None) \
		def SOS2W(dat, phis,weights=None,shift_n=0,deg=None,thresh=10,DoPrint=True) \
		def SOS2WT(dat, phis,weights=None,fil=None,shift_n=0,deg=None,thresh=10) \
		def SOS2_check(dat,qs,phis,trange='All',lim=None,calc2=False) \
		def SOS2_Fine(dat, qs, phis, shiftLim= (0,360), step=10,trange='All', lim=None) \
		def SOS2Int(data,qs,phisln,shift_n=0) \
		def Snint(nn, data, as, phisln, shift _n=0, method='trap') \
		def SnintT(nn,data, qs,phisln, shift_n=0, method='trap") \
		def Sn_check(nn,dat, qs,phis,trange='All', lim=None, shifts='All') \
		def SnFit(nn,data, qs, phisln,lam, shift_n=0) \
		def SnFitT (n,data1, qs, phisln, lam,shift_n=0) \
		def StandErr(A,axis=None)

## binningToolsErr.py
		def ReBin(qs,image, binSize=0.01, binNum=None, binRange=None)
		def ReBinOLD(qs,image,binSize=0.01,binNum=None,binRange=None)
		def BinStat(x,y, n,yerr=None,binByPoints=True, showplot=False,set_bins=None,count=True, xstat=True)
		def BinStatbyAxis(x,y,n,yerr=None,binByPoints=True,showplot=False,set_bins=None,count=True,xstat=True)
		def BinlnQ(x,y,edges,yerr=None)
		def ErrProp(B,axis=None)
		def StandErr(A,axis=None)
		def MakeEqualBins (x,n, showplot=False)
		def BinnedMean(x,y,n, binByPoints=True, showplot=False,set_ bins=None,count=False)
		def BinnedMeanCake(x,y,n,binByPoints=True,showplot=False,set_bins=None,count=False)
		def BinnedMeanByAxis(x,y,showplot=False,count=False)
		def BinnedMeanCakeByAxis(x,y,count=False)
		def rebin(dat,factor,ax=0, ends='left', truncate=False)
		def rebin_npz(fname,nt=1,nq=1,goodq=None,ends=True,scale=True,solv_per_solute=1694,solv_peak=40.58,isocorr=True):#63.9)

## DiffBinFns.py
		def MakeScanAx(paramDict,outDict,tt_corrNew=None)
		def DarkSubtract (paramDict, outDict)
		def EnergyCorr(paramDict,outDict)
		def DetectorNonlinCorr(paramDict,outDict)
		def NormalFactor (paramDict, outDict)
		def doDifference (paramDict, outDict)
		def AveAlIShots (paramDict, outDict)
		def doTimeBinning (paramDict, outDict)

## diffSignalTools.py
		def high_normalization_factor(dat, qs, qlow, qhigh)
		def norm_factor_phis(dat, qs, qlow, qhigh)
		def divide_anyshape (lg, sm)
		def DifferenceSignal (dat, nor,f_good,f_lon,f_loff, n, offset=0, totaloff=1)
		def DifferenceError(dat, nor, derrf_good, f_lon, f_loff, n, offset=0, totaloff=1)

## filterTools.py
		def FitOffset(intens1, intens2)
		def slice_histogram (dat, fil,threshold,res=100,showplot=False,field='unknown', fig=None, sub='111')
		def correlation_filter (xin,yin, fil, xray_on,threshold, showplot=False)

## fitfuns.py
		def fit_plot(func,,y, pO=None,labels=None)
		def inst_response (timeaxis, fwhm, showplot=False)
		def sequential_kinetic(xx,tO,s,t1,t2,a1,a2,return_pops=False)
		def sequential_kinetic_numeric(xx,to,s,t1,t2,a1,a2,return_pops=False)
		def inst_response_asym(timeaxis, fwhm, m, showplot=False)
		def conv_exp_decay(x,s,t, m, scale)
		def conv_exp_grow(x,s,t,m,scale)
		def print_params (popt, labels)
		def R_squared (funct, data, ydata, popt)
		def expfit(x,a, b,c, d)

## GeneralTools.py
		def find nearest(array, value)
		def divAny (lg, sml, axis=None)
		def ErrPropS(B,axis=None)
		def ErrPropM(A, B, qq, axis=None)
		def WAve(A,B,axis=None)
		def StandErr(A, axis=None)
		def ShiftCalc (bins, shift_ang)
		def TwoTheta2q (tthet, lam)
		def q2twoTheta (q, lam)

## IRFtools.py
		def IRFprep(ddata, qs=None,ts=None,ref=None, tref=None, qCenter=1.5, q Width=.15
		def chi2 (CalcSignal, ExpSignal, sigma, N, p)
		def likelihood(chi2)
		def fitIRFttO (S2N, ErrN, Ref2, ttOL, sigma, tnew1,goodt)
		def gaus(x, a, b, c, d)
		def dolRFfit(S2s,S2_std,qs=None,ts=None.ref=None.tref=None.ttOL=None,siamaL=None

## plottingTools.py
		def norm_to_point(qs, curve _ref curve_ scale, qval)
		def plot_cake (cake, qs=None, thetas=None, fig=None, sub=111, c= True)
		def plot_bow(,ys,fig=None, sub='111")
		def plot_bow_offset(x,ys,times=None, fig=None, sub='111')
		def plot_2d (ys,t=None, q=None, fig=None, sub='111', cb=True, logscan=False, ccmap='rainbow')
		def plot_2d_v2(ys,t=None,q=None,fig=None,sub='111',cb=True,logscan=False,ccmap='rainbow')
		def plot_timetrace (times, spectrum, Es,E,n_mean=3,n_bin=0, binByPoints=True, binByAxis=False, showplot=True)
		def plot_slice (ts, qs, diff, slice_plot,ax=None, logscan=False)
		def closefig (figure=")
		def closeallfigs()
		def image_from_array (arr, det=None, evt=0,scale=1, plotit=True, mapper=None,xyz=None, fig=None sub=(1,1,1),cb=True)

## ReduceFns.py
	def doAnisotropy (paramDict, outDict)
	def saveDictionary(outpath,paramDict,outDict)
	def ReduceData (inDir, exper, runs, outDir, paramDict, varDict)
	def StackProccessed(inpath,exper,runs,method='bincount')

## SetUpFns.py
	def LoadH5(fname, varDict, paramDict, outDict)
	def AzavStdFilter (paramDict, outDict)
	def MaskAzav (paramDict, outDict, listBinlnd=None)
	def setupFilters (paramDict, outDict)
	def IscatFilters (paramDict, outDict)
	def laserFilter (paramDict, outDict)
	def eBeamFilter(paramDict,outDict)
	def TTfilter (paramDict, outDict)
	def Enforcelso (paramDict, outDict)
	def saveReduction(outDir,outDict)

## SVDTools.py
	def do_svd(qs,ts, data, n=5, smooth=None, showplot=True, fig=None,
	sub=None, logscan=False)
	def do_svd_protected (qs,ts, data, n=5, smooth=None, showplot=True,
	fig=None, sub=None, logscan=False)
	def Ldiv(a,b,info=False)




