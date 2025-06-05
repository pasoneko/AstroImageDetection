import os
import yaml
import re 

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import PillowWriter
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib import patches


import healpy as hp
from reproject import reproject_from_healpix
import json

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.wcs import utils as asutils
from astropy.visualization.wcsaxes.frame import EllipticalFrame

from scipy.optimize import curve_fit
from skimage.filters import difference_of_gaussians, window, gaussian
from skimage.io import imread, imshow
from skimage.feature import peak_local_max
from skimage import color, exposure, transform
from skimage.feature import blob_dog, blob_log, blob_doh

from scipy.ndimage import gaussian_filter

import argparse as ap
from helpers import *
milagrotextcolor, milagro = setupMilagroColormap(-3, 15, 2, 256)
milagrotextcolor2, milagro2 = setupMilagroColormap(0.2, 1, 2, 256)

parser = ap.ArgumentParser(
    description="Produce a list of seed sources.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-M", "--map", default=ap.SUPPRESS, required=True, help="Significance Map file in root format"
)
parser.add_argument("--ROI-center",action="store",required=True,dest="roiCenter",type=float,nargs=2,default=None,help="ROI Center of the image (ra, dec)",)
parser.add_argument("--coordsys",action="store",dest="coordsys",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--size",action="store",dest="size",type=float,nargs=2,default=(5, 5),help="ROI Size for the image(Default: 5 x 5 degrees)",)
parser.add_argument("--plot-debug",action="store",dest="plotdebug",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plot4HWC",action="store",dest="plot4hwc",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotlhaaso",action="store",dest="plot4hwc",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotHGPS",action="store",dest="plot4hwc",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotFermi",action="store",dest="plot4hwc",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotSNR",action="store",dest="plot4hwc",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotPulsar",action="store",dest="plot4hwc",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--psfsize",action="store",dest="psfsize",type=float,nargs=1,default=0.2,help="ROI Size for the image(Default: 5 x 5 degrees)",)
parser.add_argument("--plotSurface",action="store",dest="plotsurface",default='G',help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)

args = parser.parse_args()
map_tree = args.map
roi_ra, roi_dec = args.roiCenter[0], args.roiCenter[1]
# Load the Simulation FITS file
filename = map_tree

ra_center = roi_ra
dec_center = roi_dec

print(f"ROI center {ra_center}, {dec_center}")

coord_sys = args.coordsys
xlength=args.size[0]
ylength=args.size[1]

origin = [ra_center, dec_center, xlength, ylength] 
ra_center = "{:.2f}".format(ra_center)
dec_center = "{:.2f}".format(dec_center)

array, footprint, wcs = loadmap(filename, coord_sys, origin, 'origin')
xnum = array.shape[1]
ynum = array.shape[0]

pixel_size = wcs.wcs.cdelt[1]
print(f'Degrees per pixel: {pixel_size} ')

print(f'Shape of Map in pixel number: {xnum} X {ynum}')
fig = plt.figure(figsize=(20,8))
ax = plt.subplot(1,1,1, projection=wcs)
im=ax.imshow(array,  cmap=milagro, vmin=-5, vmax=15)
fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04, label='Significance')
levels = [5, 6, 7]
hi_transform = ax.get_transform(wcs)
ax.contour(array, levels=levels, transform=hi_transform, colors='black')
plot_ax_label(ax, coord_sys)
# plot_4hwc1D(ax, wcs)
ax.set_title('HAWC Sky Map')
plt.xlim(0, xnum)
plt.ylim(0, ynum)
plt.show()