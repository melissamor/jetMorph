__author__ = 'Melissa Elizabeth Morris'
__version__ = '0.0.0'

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from scipy import optimize
from glob import glob
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
#import ellipses as el
from matplotlib.patches import Ellipse


'''
Utility Functions
'''

# Circle fitting functions, adapted from following:
#   https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
def calc_R(x,y,xc, yc):
    """
    calculate the distance of each 2D point from the center (xc, yc)
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c,x,y):
    """
    calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x,y,*c)
    return Ri - Ri.mean()

def fitcircle(x,y):
    '''
    fits a circle to given x and y points
    '''
    center_estimate = np.mean(x),np.mean(y)
    result = optimize.least_squares(f_2, center_estimate, loss="cauchy",args=[x,y])
    center_2 = result.x
    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2,x,y)
    R_2        = Ri_2.mean()
    residu_2   = Ri_2.std()
    return center_2,R_2,residu_2

# saves parameterized x, y, and t(heta) coordinates for a circle
def circleplot(xc,yc,R):
    t = np.linspace(0,2*np.pi,500)
    return [R*np.cos(t)+xc,R*np.sin(t)+yc,t]

def estimateRMS(radioImg):
    '''
    radioImg must be a 2D numpy array of the radio data
    
    masks inner 50% of data (i.e. where the source should be)
          and measures RMS outside of that

    returns mean pixel value and RMS of the unmasked region
    '''
    # define parameters used to create mask
    shape = radioImg.shape
    xo = int(shape[0]/4)
    yo = int(shape[1]/4)
    
    # create a mask for the region where the source is
    zeros = np.zeros(shape)
    zeros[xo:shape[0]-xo,yo:shape[1]-yo] = np.nan
    
    # creates masked array of the image where the image itself is masked
    maskedImg = np.ma.array(radioImg,mask=zeros)
    return np.mean(maskedImg),np.std(maskedImg)

# defines gaussian
def gaussian(x,A1,mean1,std1,upydowny):
    return A1*np.exp(-.5*((x-mean1)/std1)**2)+upydowny

# defines a line perpendicular to lines created by two points
def find_perp(xc,yc,x1,y1,x2,y2,lineLen = 25):
    '''
    xc, yc : pix coords for central point that perpendicular line will go through
    x1, y1 & x2, y2 : coords for original line
    lineLen : how long the resulting line is in pixels
    '''
    a = np.vstack([[x1,x2],np.ones(2)]).T
    result = np.linalg.lstsq(a,[y1,y2])
    slope = result[0][0]
    inte = result[0][1]
    pslope = -1./slope
    pinte = xc/slope+yc
    xdif = np.cos(np.arctan(pslope))*lineLen
    xdum = np.linspace(xc-xdif,xc+xdif,101)
    ydum = xdum*pslope+pinte
    return xdum,ydum,pslope,pinte

'''
Usable Functions
'''

# THE fitting/plotting function
def performFullFit(fileName,bdRA=None,bdDec=None,zoom=None,sigmaCut=3.,
    maskRadius=False,returnImg=False,savefig=False,uuu=0):
    '''
    Finds parameters of circle that best traces emission.

    Parameters
    ----------
    fileName : str
        path to the fits file of the radio data.
        NOTE: check how fits file is formatted and how this function
        loads them -- this is not generalized!
    bdRA, bdDec : float, optional
        RA, Dec of AGN host, in degrees. REQUIRED if maskRadius or returnImg
        are True.
    zoom : int, optional
        If specified, creates a cutout of size zoom x zoom. Makes plotting
        easier if initial image covers wide area. zoom is in pixel units.
        Good value is 250.
    sigmaCut : float, optional
        Used to calculate the pixel value above which radio emission must
        fall to be considered in fitting routine. That value is calculated by:
        mean image noise value + sigmaCut * rms noise of image.
    maskRadius : float, int, bool, optional
        The value, in pixel units, of the radius beyond which radio emission is
        mased. Set to False if considering all sigmaCut-sigma emission in img.
    returnImg : bool, optional
        if True, makes plot of image and fitted circle
    savefig : str, bool, optional
        path to where the image will be saved, set to False if you don't want
        to save image. Only works if returnImg is True.
    uuu : int, optional
        If above 1, will change how image is loaded. Use if you run into issues.

    Returns
    ----------
    center_cir : 
        x, y location of circle center in pixel units (?)
    R_cir : float
        Radius of fitted circle in pixel units
    residu_cir
        Residuals of fit
    [xhost,yhost]
        Coordinates of host galaxy in pixel units
    radioimg : numpy array
        Radio image loaded by file
    radiowcs : wcs object
        Radio wcs for loaded image
    '''

    # loads radio file
    radiohdu = fits.open(fileName)
    if uuu==0:
        radioimg = radiohdu[0].data[0][0]
    elif uuu==1:
        radioimg = radiohdu[0].data
    radiowcs = WCS(radiohdu[0].header,naxis=2)  # saves radio coordinate system (for plotting)

    if zoom:
        # create a cutout around the bent double if the image itself is huge
        radiozoom = Cutout2D(radioimg,radiowcs.wcs_world2pix(bdRA,bdDec,1),(zoom,zoom),wcs=radiowcs)
        radioimg = radiozoom.data
        radiowcs = radiozoom.wcs

    if returnImg:
        # create figure with wcs of the radio image
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111,projection=radiowcs)
        
        # show radio data
        ax1.imshow(radioimg)

    
    # locates host galaxy (NEED OPTICAL COORDINATES FOR THIS -- NOT COORDS OF RADIO SOURCE)
    if bdRA and bdDec:
        xhost,yhost=SkyCoord(bdRA*u.deg,bdDec*u.deg).to_pixel(radiowcs)
        
        if returnImg:
            # plots a red x at location of host
            ax1.plot(xhost,yhost,'rx')
    
    # estimate the RMS of the image using the outer regions
    mean,rms = estimateRMS(radioimg)

    # define the level at which to consider emission
    lvls = [mean+rms*sigmaCut]
    if returnImg:
        # plot contours
        ax1.contour(radioimg,levels=lvls,colors='white')
    # save x and y coordinates of all pixels that fall above defined levels
    y,x = np.where(radioimg > lvls)

    if maskRadius:
        # Locates all pixels above defined levels within maskRadius of the host
        dist = np.sqrt((x-xhost)**2+(y-yhost)**2)
        x = x[dist<maskRadius]
        y = y[dist<maskRadius]

        if returnImg:
            # plots circle outside of which emission is masked
            cir = plt.Circle([xhost,yhost],maskRadius,edgecolor='red',fill=False,ls='--')
            ax1.add_artist(cir)
    
    # fits circle to selected emission
    center_cir,R_cir,R_unc = fitcircle(x,y)
    
    print('Circle Radius (pixel units): {radius}'.format(radius=R_cir))
    

    if returnImg: # returns radioimg and wcs if you want to plot outside of function!
        cir = plt.Circle(center_cir,R_cir,edgecolor='pink',fill=False)
        ax1.add_artist(cir)

        # bad coding practices that get the job done hehehe (hides labels)
        ax1.set_ylabel('.',fontsize=1)
        ax1.set_xlabel('.',fontsize=1)

        if savefig:
            plt.savefig(savefig,bbox_inches='tight')

        plt.show()

    return center_cir,R_cir,R_unc,[xhost,yhost],radioimg,radiowcs

def reparamCircle(xCirc,yCirc,xhost,yhost,tCirc):
    '''
    Reparameterizes circle so galaxy host is at 0 degrees.

    Helpful when measuring jet thicknesses to do this first.

    Parameters
    ----------
    xCirc, yCirc : np.ndarray, np.ndarray
        x and y coords for the circle, usually output of circleplot
    xhost, yhost : float, float
        Pixel coordinates of AGN host
    tCirc : np.ndarray
        array of angles used to generate circle, output of circleplot
    
    Returns
    ----------
    ttCirc : np.ndarray
        new array of angles that can describe circle
    '''
    # distance from circle points to core
    cDist = np.sqrt((xCirc-xhost)**2+(yCirc-yhost)**2)
    # finds current angle of the core (where distance above is smallest)
    tCore = tCirc[cDist == np.min(cDist)]

    # generates new list of angles, abs value doesn't exceed 2pi
    ttCirc = tCirc - tCore
    ttCirc[np.where(ttCirc > np.pi)] = ttCirc[np.where(ttCirc > np.pi)]-2*np.pi
    ttCirc[np.where(ttCirc < -np.pi)] = ttCirc[np.where(ttCirc < -np.pi)]+2*np.pi
    return ttCirc

def calculate_jet_thickness(near_jet,image,circDat,plot_results=True,lineLen=False):
    '''
    Calculates thickness of jet at one location around the fitted circle.
    
    Parameters
    ----------
    near_jet : int
        index of the angle where the thickness will be measured in the list of
            angles that parameterize fitted circle
    image : 2D np.ndarray
        image array
    circDat : 2D np.ndarray
        x and y coordinates of circle
    plot_results : bool
        if True, plots jet thickness profile along the line
    lineLen : bool
        if defined, uses a line of this length (in pixel units) to measure jet thickness

    Returns
    ----------
    popt : np.ndarray
        fit matrix for gaussian
    popc : np.ndarray
        covariance matrix for gaussian
    line_dist: np.ndarray
        distances along line at which the flux was measured
    along_line: np.ndarray
        fluxes measured along line
    xd,yd: np.narray
        pixel coordinates for the line
    '''
    x1 = circDat[:,1][near_jet+1]
    y1 = circDat[:,0][near_jet+1]
    x2 = circDat[:,1][near_jet-1]
    y2 = circDat[:,0][near_jet-1]
    
    if lineLen:
        xd,yd,m,b = find_perp(circDat[:,1][near_jet],circDat[:,0][near_jet],x1,y1,x2,y2,lineLen=lineLen)
    else:
        xd,yd,m,b = find_perp(circDat[:,1][near_jet],circDat[:,0][near_jet],x1,y1,x2,y2)

    # Makes a list of the x and y positions of pixels that are very close to the line
    #    from find_perp. Also includes the pixel values and position in list of point
    #    on the perpendicular line that they are closest to.
    along_line = []
    

    # loops through all points on the line
    for i,xl in enumerate(xd):
        yl = yd[i]
        # generate a list of indices for the radioimg
        xi,yi = np.indices(np.shape(image))
        
        # create a 2d matrix of distances of pixels to the point on the line
        dist = np.sqrt((xi-xl)**2+(yi-yl)**2)
        
        # identify pixels along the line
        x,y = np.where(dist<.5)
        
        if len(x) > 0:
            along_line.append([x,y,image[y,x][0],i])
    
    along_line = np.array(along_line)
    along_line = along_line[along_line[:,3].argsort()]
    # Normalizes line such that the profile will be plotted as a function of distance from
    #    one end (which is, in this case, wherever the line starts)
    tasti = [int(i) for i in along_line[:,3]]
    nxd = xd[tasti]-xd[tasti][0]
    nyd = yd[tasti]-yd[tasti][0]

    norm_x = circDat[:,1][near_jet]-xd[tasti][0]
    norm_y = circDat[:,0][near_jet]-yd[tasti][0]
    norm_dist = (norm_x**2+norm_y**2)**.5

    line_dist = (nxd**2+nyd**2)**.5

    amp_guess = along_line[:,2][np.where(abs(line_dist-norm_dist)==min(abs(line_dist-norm_dist)))[0][0]]

    try:
        popt,popc = optimize.curve_fit(gaussian,line_dist,along_line[:,2],p0=[amp_guess,norm_dist,10,0])
    except:
        popt,popc = [-9999.,-9999.,-9999.,-9999.],[-9999.,-9999.,-9999.,-9999.]

    
    if plot_results:
        plt.figure()
        plt.plot(line_dist,along_line[:,2])
        plt.plot(line_dist,gaussian((nxd**2+nyd**2)**.5,*popt))
        plt.axvline(popt[1],color='black')
        plt.plot([popt[1]+popt[2],popt[1]-popt[2]],[popt[0]/2,popt[0]/2],color='black')
        plt.ylim(min(along_line[:,2]),max(along_line[:,2])+max(along_line[:,2])*.1)
        plt.xlim(0,max(line_dist))
        plt.show()
    
    return popt,popc,line_dist,along_line,xd,yd

def calculateExtent(fileName, bdRA, bdDec, sigmaCut=3., percentEmission=[.9,.95], display=True, zoom=False, uuu=0):
    '''
    Calculates the radius from bdRA and bdDec in which percentEmission*100 %
        of the emission above sigmaCut-sigma is contained

    Parameters
    ----------
    fileName : str
        path to image file
    bdRA : float
        RA (in deg) of host galaxy
    bdDec : float
        Dec (in deg) of host galaxy
    sigmaCut : float
        sigma threshold above which emission is considered when calculating the extent
    percentEmission : float
        the fraction of jet emission contained within the circle
            (i.e. .9 means the resulting radius will be that which
             contains 90% of the total flux)
        NOTE: can be single value or list! If a list, will return extents for each
            input value!
    display : bool
        if True, plots the radio image and the circle describing the emission
    zoom : bool
        if True, zooms in on image

    Returns
    ----------
    extent : float
        Calculates the extent of the jets in units of image pixels
    '''

    # loads radio data and wcs
    radiohdu = fits.open(fileName)
    if uuu==0:
        radioimg = radiohdu[0].data[0][0]
    elif uuu==1:
        radioimg = radiohdu[0].data
    radiowcs = WCS(radiohdu[0].header,naxis=2)

    if zoom:
        # create a cutout if the image itself is huge
        radiozoom = Cutout2D(radioimg,radiowcs.wcs_world2pix(bdRA,bdDec,1),(250,250),wcs=radiowcs)
        radioimg = radiozoom.data
        radiowcs = radiozoom.wcs

    if display:
        # create figure with wcs of the radio image
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111,projection=radiowcs)
        ax1.imshow(radioimg)

    # locates host galaxy (NEED OPTICAL COORDINATES FOR THIS -- NOT COORDS OF RADIO SOURCE)
    if bdRA and bdDec:
        xhost,yhost=SkyCoord(bdRA*u.deg,bdDec*u.deg).to_pixel(radiowcs)
        if display:
            ax1.plot(xhost,yhost,'rx')

    # estimate the RMS of the image using the outer regions
    mean,rms = estimateRMS(radioimg)

    # define the level at which to consider emission
    lvls = [mean+rms*sigmaCut]
    y,x = np.where(radioimg > lvls)

    # calculate the SNR of the x-sigma emission
    sigFlux = np.sum(radioimg[y,x])
    noiseFlux = len(x)*rms
    SNRtot = sigFlux/noiseFlux

    profile = [] # will be the measured jet emission in a range of annuli
    cprofile  = [0] # cumulative jet emission profile
    annuli = np.linspace(0,100,41) # define the annuli that will be used to measure flux
    c = 0
    for A,maskRadius in enumerate(annuli[:-1]):
        # THIS IS WHERE I AM DOING TWEAKING
        dist = np.sqrt((x-xhost)**2+(y-yhost)**2)
        ax = x[(dist>maskRadius)&(dist<annuli[A+1])]
        ay = y[(dist>maskRadius)&(dist<annuli[A+1])]
        c += len(ax)
        profile.append(np.sum(radioimg[ay,ax])/len(ax)/rms)
        cprofile.append(cprofile[-1]+np.sum(radioimg[ay,ax])-len(ax)*rms)

    # normalizes cumulative profile
    cprofile = cprofile/cprofile[-1]
    
    # saves the output as the radius value that corresponds roughly to 
    #   the point at which the cumulative profile reaches percentEmission
    if type(percentEmission) == int or type(percentEmission) == float:
        extent = annuli[np.where(cprofile < percentEmission)[0][-1]]
    else:
        extent = []
        for pE in percentEmission:
            extent.append(annuli[np.where(cprofile < pE)[0][-1]])

    if display:
        print('{sig}-sigma emission SNR: {val}'.format(sig=sigmaCut,val=SNRtot))
        ax1.contour(radioimg,levels=lvls,colors='white')
        if type(percentEmission) == int or type(percentEmission) == float:
            cir = plt.Circle([xhost,yhost],extent,edgecolor='red',fill=False,ls='--')
            ax1.add_artist(cir)
        else:
            for e in extent:
                cir = plt.Circle([xhost,yhost],e,edgecolor='red',fill=False,ls='--')
                ax1.add_artist(cir)
        plt.show()

        if type(percentEmission) == int or type(percentEmission) == float:
            plt.plot(annuli[:-1],cprofile[1:],color='black')
            plt.axhline(percentEmission,color='pink')
            plt.axvline(extent,color='pink')
            plt.show()
        else:
            plt.plot(annuli[:-1],cprofile[1:],color='black')
            for pE,e in np.array([percentEmission,extent]).T:
                plt.axhline(pE)
                plt.axvline(e)
            plt.show()
    return extent
