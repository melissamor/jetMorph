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
import ellipses as el
from matplotlib.patches import Ellipse

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
    residu_2   = sum((Ri_2 - R_2)**2)
    return center_2,R_2,residu_2

# saves parameterized x, y, and t(heta) coordinates for a circle
def circleplot(xc,yc,R):
    t = np.linspace(0,2*np.pi,500)
    return [R*np.cos(t)+xc,R*np.sin(t)+yc,t]

# ellipse fitting function, do not trust just yet (add link later)
def fitEllipse2(x,y):
    elfit2 = el.LSqEllipse()
    elfit2.fit([x,y])
    center2, width2, height2, phi2 = elfit2.parameters()
    return width2,height2,center2,phi2
    
# saves parameterized x, y, and t coordinates for an ellipse
def ellipseplot(xc,yc,a,b,phi):
    t = np.linspace(0,2*np.pi,500)
    R_rot = np.array([[np.cos(phi) , -np.sin(phi)],[np.sin(phi) , np.cos(phi)]])
    Ell = np.array([a*np.cos(t),b*np.sin(t)])
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    return [xc+Ell_rot[0,:] , yc+Ell_rot[1,:],t]

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

# THE fitting/plotting function
def performFullFit(fileName,bdRA=None,bdDec=None,zoom=None,sigmaCut=3.,
    maskRadius=False,returnImg=False,ellipse=False,savefig=False):
    
    '''
    fileName : path to the fits file of the radio data. NOTE: loading the file is not generalizable!
    bdRA : RA of bent double host, in degrees
    bdDec : Dec of bent double host, in degrees
    zoom : allows option to zoom into the image around where the bent AGN is
    sigmaCut : defines the sigma value above which radio emission must fall to be considered in the fit
    maskRadius : the value, in pixel units, of the masking radius. Set to False if no masking radius.
    returnImg : i forgot to implement this oops
    ellipse : if True, fits ellipse as well as circle
    savefig : path to where the image will be saved, set to False if you don't want to save image
    '''

    # loads radio file
    radiohdu = fits.open(fileName)
    radioimg = radiohdu[0].data[0][0]  # saves radio data
    radiowcs = WCS(radiohdu[0].header,naxis=2)  # saves radio coordinate system (for plotting)

    if zoom:
        # create a cutout around the bent double if the image itself is huge
        radiozoom = Cutout2D(radioimg,radiowcs.wcs_world2pix(bdRA,bdDec,1),(250,250),wcs=radiowcs)
        radioimg = radiozoom.data
        radiowcs = radiozoom.wcs

    # create figure with wcs of the radio image
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111,projection=radiowcs)
    
    # show radio data
    ax1.imshow(radioimg)

    
    # locates host galaxy (NEED OPTICAL COORDINATES FOR THIS -- NOT COORDS OF RADIO SOURCE)
    if bdRA and bdDec:
        xhost,yhost=SkyCoord(bdRA*u.deg,bdDec*u.deg).to_pixel(radiowcs)
        # plots a red x at location of host
        ax1.plot(xhost,yhost,'rx')
    
    # estimate the RMS of the image using the outer regions
    mean,rms = estimateRMS(radioimg)

    # define the level at which to consider emission
    lvls = [mean+rms*sigmaCut]
    # plot contours
    ax1.contour(radioimg,levels=lvls,colors='white')
    # save x and y coordinates of all pixels that fall above defined levels
    y,x = np.where(radioimg > lvls)

    if maskRadius:
        # Locates all pixels above defined levels within maskRadius of the host
        dist = np.sqrt((x-xhost)**2+(y-yhost)**2)
        x = x[dist<maskRadius]
        y = y[dist<maskRadius]

        # plots circle outside of which emission is masked
        cir = plt.Circle([xhost,yhost],maskRadius,edgecolor='red',fill=False,ls='--')
        ax1.add_artist(cir)
    
    # plot fitted circle
    center_cir,R_cir,residu_cir = fitcircle(x,y)
    cir = plt.Circle(center_cir,R_cir,edgecolor='pink',fill=False)
    ax1.add_artist(cir)

    # if specified, plot fitted ellipse
    if ellipse:
        width2,height2,center2,phi2 = fitEllipse2(x,y)
        ellipse2 = Ellipse(xy=center2, width=2*width2, height=2*height2, angle=np.rad2deg(phi2),
                               edgecolor='lime', fc='None', lw=1, label='Fit', zorder = 2)
        ax1.add_patch(ellipse2)
    
    print('Circle: {radius}'.format(radius=R_cir))
    if ellipse:
        print('Ellipse: {width}'.format(width=width2))
    
    # bad coding practices that get the job done hehehe
    ax1.set_ylabel('.',fontsize=1)
    ax1.set_xlabel('.',fontsize=1)

    if savefig:
        plt.savefig(savefig,bbox_inches='tight')

    plt.show()
    if returnImg: # make the plotting of results optional!
        return center_cir,R_cir,residu_cir,[xhost,yhost],radioimg,radiowcs
    else:
        return center_cir,R_cir,residu_cir,[xhost,yhost]

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


def calculate_jet_thickness(near_jet,image,circDat,plot_results=True,theway = True,lineLen=False):
    '''
    near_jet : 
    image : 2D image array
    circDat : 
    plot_results : plots jet thickness profile along the line
    theway : keep as true
    lineLen : if defined, uses a line of this length (in pixel units) to measure jet thickness
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
    
    if theway:
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
        
    else:
        # loops through every pixel
        for y,i in enumerate(image):
            for x,j in enumerate(i):
                # measures the distance between a pixel and every point on the line
                dist = ((x-xd)**2+(y-yd)**2)**.5
                if min(dist) < .5: # if the minimum distance is less than half a pixel
                    # finds the index (along the line) that a pixel corresponds to
                    xdyd = int(np.where(min(dist)==dist)[0])
                    # adds the line location (xdyd), pixel location in the image, and pixel value to along_line
                    along_line.append([x,y,j,int(xdyd)])
    
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

def calculateExtent(fileName, bdRA, bdDec, sigmaCut=3., percentEmission=[.9,.95], display=True, zoom=False):
    '''
    fileName : path to image file
    bdRA : RA (in deg) of host galaxy
    bdDec : Dec (in deg) of host galaxy
    sigmaCut : sigma threshold above which emission is considered when calculating the extent
    percentEmission : the fraction of jet emission contained within the circle
                        (i.e. .9 means the resulting radius will be that which
                         contains 90% of the total flux)
                    NOTE: can be single value or list! If a list, will return extents for each
                        input value!
    display : if True, plots the radio image and the circle describing the emission
    zoom : if True, zooms in on image
    '''

    # loads radio data and wcs
    radiohdu = fits.open(fileName)
    radioimg = radiohdu[0].data[0][0]
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