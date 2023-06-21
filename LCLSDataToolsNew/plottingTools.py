import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import time
import warnings
# from LCLSDataToolsNew.binningTools import *
from LCLSDataToolsNew.SVDTools import *
from cycler import cycler
able_to_pdf=True
try:
    from PyPDF2 import PdfFileReader, PdfFileWriter, PdfFileMerger
except:
    able_to_pdf=False

def norm_to_point(qs,curve_ref,curve_scale,qval):
    '''Return curve_scale normalized to share the same value of curve_ref at qval (point, as float, or range to average, as list of 2 floats).'''
    
    try:
        if len(qval)==2: ### either you got a list of 2 values; convert to indices
            qval[0]=np.argmin(np.abs(qs-qval[0]))
            qval[1]=np.argmin(np.abs(qs-qval[1]))
    except TypeError: ### or you got a number; convert to list of indices
        qv=qval
        qval=[0,0]
        qval[0]=np.argmin(np.abs(qs-qv))
        qval[1]=qval[0]+1
    c1_val=np.nanmean(curve_ref[qval[0]:qval[1]])
    c2_val=np.nanmean(curve_scale[qval[0]:qval[1]])
    norm_fac=c1_val/c2_val
    return curve_scale*norm_fac

def plot_cake(cake,qs=None,thetas=None,fig=None,sub=111,cb=True):
    '''Plot the cake image from cspad.azav in polar coordinates.'''
    if fig is None:
        plt.figure()
        plt.subplot(sub,polar=True)
    else:
        plt.figure(fig)
        if not isinstance(sub, str):
            try:
                plt.subplot(*sub,polar=True)
            except:
                plt.subplot(sub,polar=True)
        else:
            plt.subplot(sub,polar=True)
    if thetas is None:
        thetas=np.linspace(0,2*np.pi,cake.shape[0])
    if qs is None:
        qs=np.linspace(0,10,cake.shape[1])
    X,Y=np.meshgrid(thetas,qs)
    plt.pcolormesh(X,Y,cake.T)
    plt.clim(np.nanmin(cake),np.nanmax(cake))
    if cb:
        plt.colorbar()

def plot_bow(x,ys,fig=None,sub='111'):
    '''rainbow-style plot of data x [N] and Ys [M,N]'''
    if fig is None:
        plt.figure()
        ax = plt.axes()
    else:
        plt.figure(fig)
        if not isinstance(sub, str):
            plt.subplot(*sub)
        else:
            plt.subplot(sub)
        ax = plt.gca()
        

    colors=[plt.cm.viridis(i) for i in np.linspace(0, 1, ys.shape[0])]
    custom_cycler = (cycler(color=colors))
    ax.set_prop_cycle(custom_cycler)
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", category=matplotlib.cbook.mplDeprecation) #doesn't work How skip MatplotlibDeprecationWarning?
    #    ax.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, ys.shape[0])])
    plt.plot(x,ys.T)
    plt.xlabel('Q ($\AA^{-1}$)')
    plt.ylabel('I (arb)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
   


def plot_bow_offset(x,ys,times=None,fig=None,sub='111'):
    '''stacked rainbow-style plot of data x [N] and Ys [N,M].
    optionally add right-side y axis with approx. time values with times argument [M]'''
    spacing=np.nanmax(np.abs(ys))
    if fig is None:
        plt.figure()
    else:
        plt.figure(fig)
        if not isinstance(sub, str):
            plt.subplot(*sub)
        else:
            plt.subplot(sub)
    ax = plt.axes()
    plt.xlabel('Q ($\AA^{-1}$)')
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=matplotlib.cbook.mplDeprecation)
#         ax.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, ys.shape[1])])
    colors=[plt.cm.viridis(i) for i in np.linspace(0, 1, ys.shape[0])]
    custom_cycler = (cycler(color=colors))
    ax.set_prop_cycle(custom_cycler)
    offset=0
    for i in range(0,ys.shape[1]): #for each one, subtract spacing
        plt.plot(x,ys[:,i]-offset)
        offset += spacing
    if not (times is None):
        top,bottom=ax.get_ylim() #existing top and bottom of axes
        ax.get_yaxis().set_visible(False)
        ax2 = ax.twinx()
        #fit where 0=0 and offset=last time. Not exact when steps are uneven.
        m=(np.max(times)-np.min(times))/-offset #ratio of time steps to arbitrary steps
        #print(m)
        plt.ylim(m*top+np.min(times), m*bottom+np.min(times))
        plt.ylabel('Time (s)')

def plot_2d(t,x,ys,fig=None,sub='111',cb=True,logscan=False,ccmap='rainbow'):
    '''plot ys[N,M] on time [M] vs q [N] axes
    cb=True: add colorbar'''

    if fig is None:
        plt.figure()
    else:
        plt.figure(fig)
        if not isinstance(sub, (str,int)):
            plt.subplot(*sub)
        else:
            plt.subplot(int(sub))
            
    if logscan: 
        my_xticks = np.unique(t)
        t=np.arange(len(t))
        my_xticks=['%.4G'%n for n in my_xticks] #make them readable
        plt.xticks(t, my_xticks,rotation=45)
    ts,xs=np.meshgrid(t,x)
    try:
        plt.pcolormesh(ts,xs,ys,cmap=ccmap,shading='auto')
    except TypeError:
        plt.pcolormesh(ts,xs,ys.T,cmap=ccmap,shading='auto')
    #median above and below 0 * some number
    
    plt.clim([np.nanmin(ys),np.nanmax(ys)/1.5])
    #plt.clim(np.nanpercentile(ys, 1), np.nanpercentile(ys,99))
    #plt.clim(-4.5e-4, 4.5e-4)
#     plt.clim([np.nanmean(ys)-1.5*np.std(ys),np.nanmax(ys)+1.5*np.std(ys)])

    
    if cb:
        cbar=plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    plt.xlabel('t (s)')
    plt.ylabel('Q ($\AA^{-1}$)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))        
        

def plot_2d_v2(ys,t=None,q=None,fig=None,sub='111',cb=True,logscan=False,ccmap='rainbow'):
    '''plot ys[N,M] on time [M] vs q [N] axes; don't need t and q vectors if you don't want them
    cb=True: add colorbar'''

    if fig is None:
        plt.figure()
    else:
        plt.figure(fig)
        if not isinstance(sub, (str,int)):
            plt.subplot(*sub)
        else:
            plt.subplot(int(sub))
    
    if t is None:
        t=np.arange(0,ys.shape[1])
    if q is None:
        q=np.arange(0,ys.shape[0])
        
    if logscan: 
        my_xticks = np.unique(t)
        t=np.arange(len(t))
        my_xticks=['%.4G'%n for n in my_xticks] #make them readable
        plt.xticks(t, my_xticks,rotation=45)
    ts,qs=np.meshgrid(t,q)
    try:
        plt.pcolormesh(ts,qs,ys,cmap=ccmap,shading='auto')
    except:
        plt.pcolormesh(ts,qs,ys.T,cmap=ccmap,shading='auto')
    #median above and below 0 * some number
    
    plt.clim([np.nanmin(ys),np.nanmax(ys)/1.5])
#     plt.clim([np.nanmean(ys)-1.5*np.std(ys),np.nanmax(ys)+1.5*np.std(ys)])

    
    if cb:
        cbar=plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    plt.xlabel('t (s)')
    plt.ylabel('Q ($\AA^{-1}$)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
 








def plot_timetrace(times,spectrum,Es,E,n_mean=3,n_bin=0,binByPoints=True, binByAxis=False,showplot=True):
    '''Plot a time trace of an energy slice of a spectrum. For a 2d 'spectrum' at times 'times', select the values closest to energy 'E' out of energy axis 'Es'. Average the 'n_mean' points on either side and time-bin according to 'n_bin' and 'binByPoints' (see  BinnedMean). Plot times vs means and return them.'''
    idx = (np.abs(Es[:]-E)).argmin()
    onData=np.nanmean(spectrum[:,idx-n_mean:idx+n_mean],1)
    if binByAxis:
        ts,means=BinnedMeanByAxis(times, onData)
    else:
        ts,means=BinnedMean(times, onData,n_bin,binByPoints)
    if showplot:
        plt.figure()
        plt.plot(ts,means,'.')
        plt.xlabel('Time')
        plt.ylabel('I (arb)')
        plt.title('Time trace at %0.2f'%E)
    return ts,means
    

def plot_slice(ts,qs,diff,slice_plot,ax=None,logscan=False):
    '''plot a slice of the q axis defined by slice_plot=[lowq,highq], on axes "ax" or else create figure.'''
    lab=str(slice_plot)
    if ax is None:
        plt.figure()
        ax=plt.gca()
    if logscan: 
        my_xticks = np.unique(ts)
        ts=np.arange(len(ts))
        my_xticks=['%.4G'%n for n in my_xticks] #make them readable
        plt.xticks(ts, my_xticks,rotation=45)
    slice_plot[0]=np.argmin(np.abs(qs-slice_plot[0]))
    slice_plot[-1]=np.argmin(np.abs(qs-slice_plot[-1]))
    slice_plot=np.arange(slice_plot[0],slice_plot[-1])
    sl=np.nanmean(diff[:,slice_plot],1)
    ax.plot(ts,sl,'o',label=lab)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    return(sl)    
    

def closefig(figure=''):
    '''close a figure from interactive mode to static mode. '''
    if figure == '':
        figure=plt.gcf()
    figure.canvas.draw()
    time.sleep(0.1)
    plt.close()
    
def closeallfigs():
    '''close all figures from interactive mode to static mode. '''
    for i in plt.get_fignums():
        plt.figure(i)
        closefig()
        
        
def image_from_array(arr,det=None,evt=0,scale=1,plotit=True,mapper=None,xyz=None,fig=None,sub=(1,1,1),cb=True):
    #make a 2d image array from a pixel value array collected by det at time evt. If no det specified: supply xyz coords.
    #image will be made smaller by factor of scale. 
    #should be one-time map! If map is provided, use it; if not, return map at the end. 
    #xyz=det.coords_xyz(evt)
    arr=arr.flatten()
    if det:
        p_size=det.pixel_size(evt)
        xyz=det.coords_xyz(evt)
    else:
        p_size=109.92
        if not xyz:
            raise ValueError('Please specify a detector or xyz coordinate array.')
    binstep=p_size*scale
    Xs=xyz[0].flatten()
    Ys=xyz[1].flatten()
    xmin=Xs.min()
    xmax=Xs.max()
    ymin=Ys.min()
    ymax=Ys.max()
    x=np.arange(xmin,xmax,binstep)
    y=np.arange(ymin,ymax,binstep)
    img=np.zeros((len(x)*len(y)))#value assigned to each output bin
    fills=np.zeros((len(x)*len(y)))#number of physical pixels assigned to each bin
    if mapper is None:
        xbinned=np.digitize(Xs,x)-1
        ybinned=np.digitize(Ys,y)-1 #minus one because binning started in bin 1 instead of zero
        flatbinned=xbinned+ybinned*len(x) #1d indices over *image* indices
        mapper=flatbinned
    else:
        flatbinned=mapper
        #mapper saves only 6% of time

    for i,pixel in np.ndenumerate(arr):
        i=i[0]
        img[flatbinned[i]]+=pixel
        fills[flatbinned[i]]+=1

    out=img/fills
    
    out=np.reshape(out,(len(x),len(y)))
    if plotit:
        if fig is None:
            plt.figure()
        else:
            plt.figure(fig)
            plt.subplot(*sub)
        plt.imshow(out,interpolation="none")
        if cb:
            plt.colorbar()
    return(out,x,y)


