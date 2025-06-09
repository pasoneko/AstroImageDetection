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
current=os.getcwd()
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
parser.add_argument("--plot-debug",action="store",dest="plotdebug",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plot4HWC",action="store",dest="plot4hwc",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotlhaaso",action="store",dest="plotlhaaso",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotHGPS",action="store",dest="plothgps",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotFermi",action="store",dest="plotfermi",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotSNR",action="store",dest="plotsnr",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotPulsar",action="store",dest="plotpulsar",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--psfsize",action="store",dest="psfsize",type=float,nargs=1,default=0.2,help="ROI Size for the image(Default: 5 x 5 degrees)",)
parser.add_argument("--plotPDF",action="store",dest="plotPDF",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotSurface",action="store",dest="plotSurface",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--plotStackSignif",action="store",dest="plotStackSignif",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
# parser.add_argument("--plotPDF",action="store",dest="plotPDF",default=False,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
parser.add_argument("--colormap",action="store",dest="colormap",default=milagro,help="Image Coordinate: 'G', 'C'.  (Default: 'G')",)
# parser.add_argument( "-o", "--out-dir", dest="outdir", default="{0}".format(current), help="Directory in which to save plots.")
args = parser.parse_args()

outdir = os.getcwd()+ '/catplots/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

map_tree = args.map
roi_ra, roi_dec = args.roiCenter[0], args.roiCenter[1]
is4hwc=args.plot4hwc
ishgps=args.plothgps
islhaaso=args.plotlhaaso
isfermi=args.plotfermi
issnr=args.plotsnr
ispular=args.plotpulsar
psf=args.psfsize
catalogs=[]
if is4hwc:
    catalogs.append('4hwc')
if ishgps:
    catalogs.append('hgps')
if islhaaso:
    catalogs.append('lhaaso')
if isfermi:
    catalogs.append('fermi')
if issnr:
    catalogs.append('snr')
if ispular:
    catalogs.append('pulsar')
else:
    catalogs.append('None')

ispdf=args.plotPDF
if ispdf:
    pdf = PdfPages(os.path.join(outdir, f'run-{roi_ra}-{roi_dec}-plots.pdf')) 
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

if ispdf:
    makeplot(array, -5, 15, wcs, args.colormap, coord_sys, "Gamma-ray Sky", xnum, ynum, outdir, catalogs, pdf=True)
else:
    makeplot(array, -5, 15, wcs, args.colormap, coord_sys, "Gamma-ray Sky", xnum, ynum, outdir, catalogs)

if np.max(array) < 5:
    print(f'Algorithm wont proceed. Significance Map has no data greater than 5$\sigma$)')
    exit

# Preprocessing Step: Floor the image for analysis
print(np.min(array))
print(np.max(array))
if np.min(array) < -5:
    print("Image Floored to -5 sigma")
    array = invrelu(array, floor_min=-5)

if args.plotSurface:
    # Make the 3D Intensity Distribution
    smoothed = array
    smoothed = smoothed[::4, ::4]
    x = np.arange(smoothed.shape[1])
    y = np.arange(smoothed.shape[0])
    X, Y = np.meshgrid(x, y)

    coordinates = peak_local_max(smoothed, min_distance=5, threshold_rel=0.1)
    peak_x = coordinates[:, 1]
    peak_y = coordinates[:, 0]
    peak_z = smoothed[peak_y, peak_x]


    fig = plt.figure(figsize=(6, 5))#, dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, smoothed, cmap='plasma', edgecolor='none')
    peaks_plot = ax.scatter(peak_x, peak_y, peak_z, c='black', s=50, label='Peaks')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Surface Histogram of Significance')
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    ani.save(outdir+f'gaussian_surface_rotation-{1825}.gif', writer='pillow')

if args.plotStackSignif:
    #Make Stacked Significane Range Plots
    x = np.linspace(3, np.max(array)+3, 15)
    fig = plt.figure(figsize=(20,8))
    ax = plt.subplot(1, 1, 1, projection=wcs)
    im = ax.imshow(array, cmap=milagro, vmin=x[0], vmax=x[1])
    cbar = fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, label='Significance')
    plot_ax_label(ax, coord_sys)
    ax.set_title(f'HAWC Sky Map from {x[0]} to {x[1]}')
    ax.set_xlim(0, xnum)
    ax.set_ylim(0, ynum)

    def update(i):
        if i < len(x) - 1:
            im.set_clim(vmin=x[i], vmax=x[i+1])
            ax.set_title(f'HAWC Sky Map from {x[i]} to {x[i+1]}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(x)-1, interval=1000, blit=False)
    ani.save(outdir+f'hawc_sky_map-{1825}.gif', writer=PillowWriter(fps=1))
    plt.clf()

#Normalize the image between 0 and 1 for analysis
image = array
image = (image-np.min(image))/(np.max(image)-np.min(image))
makeplot(image, 0, 1, wcs, args.colormap, coord_sys, "Normalized Input Image", xnum, ynum, outdir, catalogs)

# Calculate the 1D histogram of the image
pixels = array.flatten()
binlen = int(len(pixels)/1000)
print(f"Length of bins={binlen}")
counts, bin_edges = np.histogram(pixels, bins=binlen, range=(-4, np.max(pixels)))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
x_exp = np.linspace(-5, 5, binlen)
y_exp = gaussian_fit(x_exp, counts.max(), 0, 1)
initial_guess = [counts.max(), bin_centers[np.argmax(counts)], np.std(pixels)]
popt, _ = curve_fit(gaussian_fit, bin_centers, counts, p0=initial_guess)
x_fit = np.linspace(-6, np.max(pixels), binlen)
y_fit = gaussian_fit(x_fit, *popt)

plt.figure(figsize=(8, 8))
log_counts = np.where(counts > 0, counts, 1) 
plt.hist(pixels, bins=binlen, range=(np.min(pixels), np.max(pixels)), color='fuchsia', edgecolor='green', alpha=0.6, label='Histogram (Counts)', histtype='step', linewidth=3)
plt.plot(x_exp, y_exp, 'red', linewidth=2, label=f'Expectation\nμ=0, σ=0.01')
plt.plot(x_fit, y_fit, 'blue', linewidth=2, label=f'Fit\nμ={popt[1]:.5f}, σ={popt[2]:.5f}')
plt.yscale('log')
plt.ylim(0.5, 1e5)
plt.xlim(-6, 6)
plt.xlabel('Pixel Intensity')
plt.ylabel('Log(Counts)')
plt.title('Histogram of Image')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(outdir+'OriginalImageHistogram.png')
plt.clf()


# Find the Difference of Gaussian (DoG) Image
print(psf/pixel_size)
new_image = difference_of_gaussians(image, 1, psf/pixel_size)#PSF of the dec band
if ispdf:
    makeplot(new_image, 0, np.max(new_image), wcs, args.colormap, coord_sys, "Gaussian Subtracted Image", xnum, ynum, outdir, catalogs, pdf=True)
else:
    makeplot(new_image, 0, np.max(new_image), wcs, args.colormap, coord_sys, "Gaussian Subtracted Image", xnum, ynum, outdir, catalogs)


# Calculate the 1D histogram of the DoG Image
pixels = new_image.flatten()
sigma_resid2 = np.std(new_image)
counts, bin_edges = np.histogram(pixels, bins=200, range=(np.min(pixels), np.max(pixels)))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
x_exp = np.linspace(np.min(bin_edges), np.max(bin_edges), 200)
y_exp = gaussian_fit(x_exp, counts.max(), 0, 0.01)

initial_guess = [counts.max(), bin_centers[np.argmax(counts)], np.std(pixels)]
popt, _ = curve_fit(gaussian_fit, bin_centers, counts, p0=initial_guess)
x_fit = np.linspace(bin_centers[0], bin_centers[-1], 500)
y_fit = gaussian_fit(x_fit, *popt)

sigma_resid = popt[2]
deviation = 3*sigma_resid
plt.figure(figsize=(10, 8))
log_counts = np.where(counts > 0, counts, 1) 
plt.hist(pixels, bins=200, range=(np.min(pixels), np.max(pixels)),  color='fuchsia', edgecolor='green', alpha=0.6, label='Histogram (Counts)',histtype='step', linewidth=3)
# plt.plot(x_exp, y_exp, 'cyan', linewidth=2, label=f'Expectation\nμ=0, σ=0.01')
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit\nμ={popt[1]:.5f}, σ={popt[2]:.5f}')
plt.axvline(deviation, label=f'3 $\sigma$  = {3*popt[2]:.5f}', color='black')
plt.axvline(popt[1], label=f'Mean = {popt[1]:.5f}')
plt.yscale('log')
plt.ylim(1, 1e8)
plt.xlim(np.min(bin_edges), np.max(bin_edges))
plt.xlabel('Pixel Intensity')
plt.ylabel('Log(Counts)')
plt.title('Histogram of Gaussian Subtracted Image')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(outdir+'DoGHistogram.png')
plt.clf()


# Find the blobs from the DoG image
print("Finding point source blobs")
ps_blobs = blob_dog(new_image, min_sigma=0.1/pixel_size, max_sigma=psf/pixel_size, threshold=0.01, exclude_border=20)#, overlap=0.9)
print("Number of point source blobs=",len(ps_blobs))
print("Finding extended source blobs")
ext_blobs = blob_dog(new_image,min_sigma=psf/pixel_size, max_sigma=0.5/pixel_size, threshold=0.01, exclude_border=20)#, overlap=0.9)
print("Number of ext source blobs=",len(ext_blobs))

# DoG Image Intensity Filtering
intensity_min = deviation
intensity_min2 = deviation
ps_filtered_blobs, ps_filtered_coords, ps_filtered_radius = blob_filter_intensity(ps_blobs, new_image, intensity_min, wcs)
print("No of point blobs after filtering = ", len(ps_filtered_blobs))
ext_filtered_blobs, ext_filtered_coords, ext_filtered_radius = blob_filter_intensity(ext_blobs, new_image, intensity_min2, wcs)
print("No of extended blobs after filtering = ", len(ext_filtered_blobs))

if ispdf:
    makeplot(new_image, 0, np.max(new_image), wcs, args.colormap, coord_sys, "Gaussian Subtracted Image", xnum, ynum, outdir, catalogs, pdf=True)
else:
    makeplot(new_image, 0, np.max(new_image), wcs, args.colormap, coord_sys, "Gaussian Subtracted Image", xnum, ynum, outdir, catalogs)

fig=plt.figure(figsize=(8, 8))
ax = plt.subplot(1,1,1, projection=wcs)
im=ax.imshow(new_image, cmap=milagro, vmin=0)
fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04, label='Significance')
ax.set_xlim(0, xnum)
ax.set_ylim(0, ynum)
ax.set_title("Seed Sources After Normalized Intensity Filtering")
plot_ax_label(ax, coord_sys)
plot_ps_blob(ax, ps_filtered_blobs, wcs)
plot_ext_blob(ax, ext_filtered_blobs, wcs)
plt.show()
plt.close(fig)