from astropy.io import fits
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import healpy as hp
from reproject import reproject_from_healpix
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import astropy.wcs.utils as astropy_utils
from astropy.io.fits import Header
import math
import tempfile
import urllib
import base64
from datetime import datetime
import json
import copy
import yaml
import os
import re
from threeML import *
from hawc_hal import HAL, HealpixConeROI, HealpixMapROI
from astropy.table import Table
from matplotlib.patches import Ellipse
import sys
sys.path.append("imagecatalog/")
#Import HESS catalog
hess_catalog = Table.read("datasets/hgps_catalog_v1.fits.gz")
hess_sourcename = hess_catalog['Source_Name']
hess_coords = SkyCoord(l=hess_catalog["GLON"], b=hess_catalog["GLAT"], frame="galactic")

#Import HAWC Catalog
hawcv4_table = pd.read_csv('datasets/4hwc.txt', delimiter=',')
hwc4_name = hawcv4_table['Name'].tolist()
hwc4_ra = hawcv4_table['Ra'].tolist()
hwc4_dec = hawcv4_table['Dec'].tolist()
hwc4_ext = hawcv4_table['Ext'].tolist()
hawcv4_coords = SkyCoord(ra=hwc4_ra, dec=hwc4_dec, unit='deg', frame='icrs')


#Import LHAASO Catalog
df = pd.read_csv("datasets/lhaaso_cat.csv", comment="#", header=None)
df.columns = [
    "Source name", "Components", "RA_2000", "Dec_2000", "Sigma_p95_stat",
    "r_39", "TS", "N0", "Gamma", "TS_100", "Association"
]

def clean_value(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '')
    if '<' in val:
        return float(val.replace('<', '').strip())
    elif 'pm' in val or '±' in val or '\\pm' in val:
        parts = re.split(r'±|\\pm|pm', val)
        try:
            return float(parts[0].strip())
        except:
            return np.nan
    try:
        return float(val)
    except:
        return val
for col in ["Source name","r_39", "N0", "Gamma"]:
    df[col] = df[col].apply(clean_value)
df["Source name"] = df["Source name"].replace(r'^\s*$', np.nan, regex=True).ffill()


#Import Fermi-LAT Catalog
fermi_fits = fits.open('datasets/gll_psc_v35.fits')
p_data = fermi_fits[1].data
fermi_fulltable = Table(p_data)

def loadmap(filename, coord_sys, coords,*args):
    print("Coords=",coords)
    with fits.open(filename) as ihdu:
        if 'xyrange' in args:
            e1, e2, e3 , e4 = coords
            cX, cY = (e1+e2)/2, (e3+e4)/2
            xR = int(np.abs(e1-e2)/(1/360))
            yR = int(np.abs(e3-e4)/(1/360))
        if 'origin' in args:
            cX, cY, xR, yR = coords
            xR = int(xR/(1/360))
            yR = int(yR/(1/360))
        print(cX, cY, xR, yR)
        if coord_sys == 'C':   ###Celestial Coordinate System
            target_header = Header()
            target_header['NAXIS'] = 2
            target_header['NAXIS1'] = xR
            target_header['NAXIS2'] = yR
            target_header['CTYPE1'] = 'RA---MOL'
            target_header['CRPIX1'] = xR/2
            target_header['CRVAL1'] = cX
            target_header['CDELT1'] = -1./360
            target_header['CUNIT1'] = 'deg     '
            target_header['CTYPE2'] = 'DEC--MOL'
            target_header['CRPIX2'] = yR/2
            target_header['CRVAL2'] = cY
            target_header['CDELT2'] = 1./360
            target_header['CUNIT2'] = 'deg     '
            target_header['COORDSYS'] = 'icrs    '
            print("A")
        if coord_sys == 'G':  ###Galactic Coordinate System
            target_header = Header()
            target_header['NAXIS'] = 2
            target_header['NAXIS1'] = xR
            target_header['NAXIS2'] = yR
            target_header['CTYPE1'] = 'GLON-AIT'
            target_header['CRPIX1'] = xR/2
            target_header['CRVAL1'] = cX
            target_header['CDELT1'] = -2./360
            target_header['CUNIT1'] = 'deg     '
            target_header['CTYPE2'] = 'GLAT-AIT'
            target_header['CRPIX2'] = yR/2
            target_header['CRVAL2'] = cY
            target_header['CDELT2'] = 2./360
            target_header['CUNIT2'] = 'deg     '
            target_header['COORDSYS'] = 'galactic    '
            print("B")
        
        skymap_data = ihdu[1].data["significance"] #.data["significance"]
        ihdu[1].header['COORDSYS'] = 'icrs    '
        wcs = WCS(target_header)
        print(wcs)
        array, footprint = reproject_from_healpix(ihdu[1],target_header)
        print("Fits File loaded")
        
    return array, footprint, wcs

def loadvgpsmap(filename, coord_sys, coords,*args):
    print("Coords=",coords)
    with fits.open(filename) as ihdu:
        if 'xyrange' in args:
            e1, e2, e3 , e4 = coords
            cX, cY = (e1+e2)/2, (e3+e4)/2
            xR = int(np.abs(e1-e2)/(1/360))
            yR = int(np.abs(e3-e4)/(1/360))
        if 'origin' in args:
            cX, cY, xR, yR = coords
            xR = int(xR/(1/360))
            yR = int(yR/(1/360))
        print(cX, cY, xR, yR)
        if coord_sys == 'C':   ###Celestial Coordinate System
            target_header = Header()
            target_header['NAXIS'] = 2
            target_header['NAXIS1'] = xR
            target_header['NAXIS2'] = yR
            target_header['CTYPE1'] = 'RA---MOL'
            target_header['CRPIX1'] = xR/2
            target_header['CRVAL1'] = cX
            target_header['CDELT1'] = -1./360
            target_header['CUNIT1'] = 'deg     '
            target_header['CTYPE2'] = 'DEC--MOL'
            target_header['CRPIX2'] = yR/2
            target_header['CRVAL2'] = cY
            target_header['CDELT2'] = 1./360
            target_header['CUNIT2'] = 'deg     '
            target_header['COORDSYS'] = 'icrs    '
            print("A")
        if coord_sys == 'G':  ###Galactic Coordinate System
            target_header = Header()
            target_header['NAXIS'] = 2
            target_header['NAXIS1'] = xR
            target_header['NAXIS2'] = yR
            target_header['CTYPE1'] = 'GLON-AIT'
            target_header['CRPIX1'] = xR/2
            target_header['CRVAL1'] = cX
            target_header['CDELT1'] = -1./360
            target_header['CUNIT1'] = 'deg     '
            target_header['CTYPE2'] = 'GLAT-AIT'
            target_header['CRPIX2'] = yR/2
            target_header['CRVAL2'] = cY
            target_header['CDELT2'] = 1./360
            target_header['CUNIT2'] = 'deg     '
            target_header['COORDSYS'] = 'galactic    '
            print("B")

        skymap_data = ihdu[1].data["SIGNAL"] #.data["significance"]
        ihdu[1].header['COORDSYS'] = 'icrs    '
        wcs = WCS(target_header)
        print(wcs)
        array, footprint = reproject_from_healpix(ihdu[1],target_header, nested=True)
        print("Fits File loaded")
        
    return array, footprint, wcs

def plot_4FGL(ax, wcs, ra_center, dec_center, xlength, ylength):
    masks = [fermi_fulltable['GLON'] >= (float(ra_center)-xlength),  fermi_fulltable['GLON'] <= float(ra_center)+xlength,  fermi_fulltable['GLAT'] >= float(dec_center)-ylength,  fermi_fulltable['GLAT'] <= float(dec_center)+ylength]
    full_mask = reduce(np.logical_and, masks)
    fermi_table = fermi_fulltable[full_mask]
    fermi_name = fermi_table['Source_Name']
    fermi_ra = fermi_table['RAJ2000']
    fermi_dec = fermi_table['DEJ2000']
    fermi_semi_major = fermi_table['Conf_68_SemiMajor']
    fermi_semi_minor = fermi_table['Conf_68_SemiMinor']
    fermi_angle = fermi_table['Conf_68_PosAng']
    fermi_l = fermi_table['GLON']
    fermi_b = fermi_table['GLAT']
    for i in range(len(fermi_name)):
        coord2 = SkyCoord(ra=fermi_ra[i]*u.deg, dec=fermi_dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
        ax.annotate(fermi_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90),
        textcoords='offset points', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
        linestyle='-'), color='cyan', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
        center = (float(pixelcoord[0]), float(pixelcoord[1]))
        fermiext = Ellipse(xy=center, width=fermi_semi_major[i]/0.0025, height=fermi_semi_minor[i]/0.0025, angle=fermi_angle[i], fc='None', ec='cyan', linewidth=2)
        ax.add_patch(fermiext)

def label_4hwc(ax, wcs):
    if len(ax.shape) == 1:
        return plot_4hwc1D(ax, wcs)
    if len(ax.shape)== 2:
        return plot_4hwc2D(ax, wcs)

def plot_4hwc1D(ax, wcs):
    try:
        for i in range(len(ax)):
            for i in range(len(hwc4_name)):
                coord2 = SkyCoord(ra=hwc4_ra[i]*u.deg, dec=hwc4_dec[i]*u.deg)
                pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
                ax[i].plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
                ax[i].annotate('4HWC '+hwc4_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
                textcoords='offset points', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
                linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
                path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
                hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/0.005, color='red', linewidth=2, fill=False, linestyle='-')
                ax[i].add_patch(hawcext)
    except:
        for i in range(len(hwc4_name)):
            coord2 = SkyCoord(ra=hwc4_ra[i]*u.deg, dec=hwc4_dec[i]*u.deg)
            pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
            ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
            ax.annotate('4HWC '+hwc4_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, -90), 
            textcoords='offset points', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
            linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
            path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/0.005, color='red', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)
        hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/0.005, color='red', linewidth=2, fill=False, linestyle='-', label='4HWC Extension')
        ax.add_patch(hawcext)

def plot_4hwc2D(ax, wcs):
    n , m = ax.shape
    for i in range(n):
        for j in range(m):
            for i in range(len(hwc4_name)):
                coord2 = SkyCoord(ra=hwc4_ra[i]*u.deg, dec=hwc4_dec[i]*u.deg)
                pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
                ax[i][j].plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
                ax[i][j].annotate('4HWC '+hwc4_name[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
                textcoords='offset points', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
                linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
                path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
                hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), hwc4_ext[i]/0.0025, color='red', linewidth=2, fill=False, linestyle='-')
                ax[i][j].add_patch(hawcext)

def invrelu(x, floor_min=-3):
	return np.maximum(floor_min, x)
def relu(x, ceil_max=15):
	return np.minimum(ceil_max, x)

def calc_norm_from_act(image, x):
    normalized_data = (x - np.min(image))/(np.max(image) - np.min(image))
    return normalized_data

def calc_act_from_norm(image, x, min, max):
    actual_data = np.min(image) + x * (np.max(image) - np.min(image))
    return actual_data


def check_circle_relation(x1, y1, r1, x2, y2, r2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if distance + r2 <= r1:
        return 0
    elif distance <= r2:
        return 1
    elif distance < 0.6 * (r1 + r2):
        return 2
    else:
        return 3

def plot_ps_blob(ax, ps_blobs, wcs):
    try:
        for i in range(len(ps_blobs)):
                    blob = ps_blobs[i]
                    y, x, r = blob
                    ax.plot(x, y, marker='x', markersize=5, color='green')
                    ax.annotate('Blob '+ str(i),xy=(x, y), xycoords='data', xytext=(100, -90), 
                    textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='green', linewidth=2, 
                    linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
                    path_effects=[pe.withStroke(linewidth=2, foreground="green")])
                    c = plt.Circle((x, y), r, color='green', linewidth=3, fill=False)
                    ax.add_patch(c)
    except:
        pass

def plot_ext_blob(ax, ext_blobs, wcs):
    try:
        for i in range(len(ext_blobs)):
                    blob = ext_blobs[i]
                    y, x, r = blob
                    ax.plot(x, y, marker='x', markersize=5, color='white')
                    ax.annotate('Ext Blob '+str(i),xy=(x, y), xycoords='data', xytext=(100, 90), 
                    textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
                    linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
                    path_effects=[pe.withStroke(linewidth=2, foreground="purple")])
                    c = plt.Circle((x, y), r, color='gray', linewidth=3, fill=False)
                    ax.add_patch(c)
    except:
        pass

def blob_filter_intensity(blobs, image, min_intensity, wcs):
    hfiltered_blobs = []
    hfiltered_coords = []
    hfiltered_radius = []
    for blob in blobs:
        y, x, r = blob
        y_min, y_max = int(max(y - r, 0)), int(min(y + r, np.array(image).shape[0]))
        x_min, x_max = int(max(x - r, 0)), int(min(x + r, np.array(image).shape[1]))
        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        distance_from_center = np.sqrt((y_grid - y)**2 + (x_grid -x)**2)
        circular_mask = distance_from_center <= r-0.2*r
        mean_intensity = np.array(image)[y_min:y_max, x_min:x_max][circular_mask].mean()
        if min_intensity <= mean_intensity :
            coord = astropy_utils.pixel_to_skycoord(x, y, wcs=wcs)
            hfiltered_blobs.append(blob)
            hfiltered_coords.append(coord.icrs)
            hfiltered_radius.append(r*(1/360))
            print(r"Blob Intensity {}, Coords ({}, {}), Radius {}, Pixel Radius {}".format(mean_intensity, coord.icrs.ra, coord.icrs.dec, r, r*(1/360)))
    return hfiltered_blobs, hfiltered_coords, hfiltered_radius


def blob_filter_overlap(hfiltered_blobs, hfiltered_coords, hfiltered_radius, hfiltered_blobs2, hfiltered_coords2, hfiltered_radius2):
    i=0
    try:
        while i <= len(hfiltered_blobs2):
            c1 = hfiltered_coords2[i]
            # print(c1)
            r1 = hfiltered_radius2[i]
            x1, y1, r1 = c1.ra.deg, c1.dec.deg, r1
            j=0
            while j<=len(hfiltered_blobs):
                c2 = hfiltered_coords[j]
                r2 = hfiltered_radius[j]
                x2, y2, r2 = c2.ra.deg, c2.dec.deg, r2
                x=check_circle_relation(x1, y1, r1, x2, y2, r2)
                # print("j",j)
                if x == 1 or x == 0 or x == 2:
                    hfiltered_blobs.pop(j)
                    hfiltered_coords.pop(j)
                    hfiltered_radius.pop(j)
                    j=j-1
                j = j+1
            i=i+1
    except:
        pass
    return hfiltered_blobs, hfiltered_coords, hfiltered_radius

def SNRCat2():
    url = "http://snrcat.physics.umanitoba.ca/SNRdownload.php?table=SNR"
    tmp = tempfile.NamedTemporaryFile()
    try:
        urllib.request.urlretrieve(url, tmp.name)
    except:
        urllib.urlretrieve(url, tmp.name)
    filename = tmp.name
    f = open(filename)
    outf = open("snrcat_data_%s.txt" % datetime.now().date(), "w")
    assocs = []
    assocs_ra = []
    assocs_dec = []
    i=0
    with open(filename) as f:
        for ln in f.readlines()[2:]:
            col_list = ln.split(';')
            try:
                name_index=col_list.index('G')
                ra_index=col_list.index('J2000_ra (hh:mm:ss)')
                dec_index=col_list.index('J2000_dec (dd:mm:ss)')
            except:
                name_index=name_index
                ra_index=ra_index
                dec_index=dec_index
            if i>1:
                snrcoord_time = SkyCoord(col_list[ra_index], col_list[dec_index], unit=(u.hourangle, u.deg),frame='icrs')
                assocs.append(col_list[name_index])
                assocs_ra.append(snrcoord_time.ra.deg)
                assocs_dec.append(snrcoord_time.dec.deg)
            i=i+1
        return assocs, assocs_ra, assocs_dec


def plot_snrcat(ax, wcs, labels=True):
    assoc, ra, dec = SNRCat2()
    i = 0
    for i in range(len(assoc)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='cyan')
        if labels:
            ax.annotate(assoc[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
            textcoords='offset points', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
            linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
            path_effects=[pe.withStroke(linewidth=2, foreground="gray")])
    ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white', label='SNR')

# source injection plotting from rishi  #Ian
def injected_sources_plot(names, ra, dec, ext, ax, wcs):
    for i in range(len(names)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='blue')
        ax.annotate('Injected '+names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', fontsize=16, xytext=(100, -90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
        linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="blue")])
        r = ext[i] / 0.0027
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color='blue', linewidth=3, fill=False)
        ax.add_patch(c)

# plotting custom sources
def custom_sources_plot(names, ra, dec, ext, ax, wcs, color, cgps=False, xray=False):
    for i in range(len(names)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
        ax.annotate('4HWC'+names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', fontsize=16, xytext=(100, 90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
        linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="red")])
        if cgps:
            r = ext[i] / 0.05
        elif xray:
            r = ext[i] / 0.004999999
        else:
            r = ext[i] / 0.0027
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color=color[i], linewidth=3, fill=False)
        ax.add_patch(c)

# plotting fermi sources
def fermi_sources_plot(names, ra, dec, ext, ax, wcs):
    for i in range(len(names)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
        ax.annotate(names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(10, -90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
        linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="blue")])
        r = ext[i] / 0.0027
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color='blue', linewidth=3, fill=False)
        ax.add_patch(c)

# pipeline fit results  #Ian
def pipeline_fit_plot(names, ra, dec, ext, ax, wcs):
    for i in range(len(names)):
        coord2 = SkyCoord(ra=ra[i]*u.deg, dec=dec[i]*u.deg)
        pixelcoord = astropy_utils.skycoord_to_pixel(coord2, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='x', markersize=5, color='white')
        ax.annotate('4HAWC '+names[i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(10, -90), 
        textcoords='offset pixels', arrowprops=dict(arrowstyle="-",color='gray', linewidth=2, 
        linestyle='-'), color='white', rotation=0, ha='right', va='center' , fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground="green")])
        r = ext[i] / 0.0027
        c = plt.Circle((pixelcoord[0], pixelcoord[1]), r, color='green', linewidth=3, fill=False)
        ax.add_patch(c)


# Function to extract RA and Dec from the filename
def extract_ra_dec(filename):
    pattern = r'model_\d+_roi_(\d+\.\d+)_(-?\d+\.\d+)\.yaml'
    match = re.search(pattern, filename)
    if match:
        ra, dec = float(match.group(1)), float(match.group(2))
        return ra, dec
    else:
        raise ValueError(f"Filename format is incorrect: {filename}")

# Function to extract RA and Dec from the filename
def extract_run(filename):
    pattern = r'model_+(\d+)+_roi_(\d+\.\d+)_(-?\d+\.\d+)\.yaml'
    match = re.search(pattern, filename)
    if match:
        run = float(match.group(1))
        return run
    else:
        raise ValueError(f"Filename format is incorrect: {filename}")

# Function to parse a YAML file and extract the required parameters
def parse_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    
    sources = []
    for source, properties in data.items():
        if 'Gaussian_on_sphere' in properties:
            lon0 = properties['Gaussian_on_sphere']['lon0']['value']
            lat0 = properties['Gaussian_on_sphere']['lat0']['value']
            sigma = properties['Gaussian_on_sphere'].get('sigma', {}).get('value', None)
            sources.append({
                'source_name': source,
                'lon0': lon0,
                'lat0': lat0,
                'sigma': sigma,
                # 'spectrum': spectrum,
                # 'index' : index
            })
        elif 'position' in properties:
            lon0 = properties['position']['ra']['value']
            lat0 = properties['position']['dec']['value']
            sigma = 0
            sources.append({
                'source_name': source,
                'lon0': lon0,
                'lat0': lat0,
                'sigma': sigma,
            })
        else:
            pass
    return sources

def plot_ax_label(ax, coord_sys):
    if coord_sys == 'C':
        ax.set_xlabel(r"$ra^o$")
        ax.set_ylabel(r"$dec^o$")
    else:
        ax.set_ylabel(r"$b^o$")
        ax.set_xlabel(r"$l^o$")

# Create a circular mask
def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a circular mask for a given height (h), width (w),
    with specified center and radius.
    """
    if center is None:
        center = (int(w / 2), int(h / 2)) 
    if radius is None:
        radius = min(h, w) / 4

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def parula_cmap():

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
    0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
    [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
    0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
    0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
    [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
    0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
    0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
    [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
    0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
    0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
    0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
    [0.0589714286, 0.6837571429, 0.7253857143], 
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
    0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
    [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
    0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
    0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
    [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
    0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
    [0.7184095238, 0.7411333333, 0.3904761905], 
    [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
    0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
    0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
    [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
    0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
    0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
    [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map

def setupMilagroColormap(amin, amax, threshold, ncolors):
    thresh = (threshold - amin) / (amax - amin)
    if threshold <= amin or threshold >= amax:
        thresh = 0.
    dthresh = 1 - thresh
    threshDict = { "blue"  : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.615*dthresh, 1, 1),
                              (thresh+0.692*dthresh, 1, 1),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "green" : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.231*dthresh, 0, 0),
                              (thresh+0.308*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 0.8, 0.8),
                              (thresh+0.462*dthresh, 1, 1),
                              (thresh+0.615*dthresh, 0.8, 0.8),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.846*dthresh, 0, 0),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "red"   : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0.5, 0.5),
                              (thresh+0.231*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 1, 1),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)) }

    newcm = mpl.colors.LinearSegmentedColormap("thresholdColormap",
                                               threshDict,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

def plot_1lhaaso(ax, wcs):
    for i in range(len(df)):
        if df['RA_2000'][i] != ' ':
            coord2 = SkyCoord(ra=float(df['RA_2000'][i])*u.deg, dec=float(df['Dec_2000'][i])*u.deg)
            coord_gal = coord2.galactic
            pixelcoord = astropy_utils.skycoord_to_pixel(coord_gal, wcs)
            ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
            ax.annotate(df['Source name'][i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 90), 
            textcoords='offset points', arrowprops=dict(arrowstyle="-",color='blue', linewidth=2, 
            linestyle='-.'), color='white', rotation=0, ha='right', va='center' ,
            path_effects=[pe.withStroke(linewidth=2, foreground="blue")])
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), float(df['r_39'][i])/0.005, color='blue', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)


def plot_hgps(ax, wcs):
    for i in range(len(hess_catalog)):
        coord2 = SkyCoord(ra=float(hess_catalog['RAJ2000'][i])*u.deg, dec=float(hess_catalog['DEJ2000'][i])*u.deg)
        coord_gal = coord2.galactic
        pixelcoord = astropy_utils.skycoord_to_pixel(coord_gal, wcs)
        ax.plot(pixelcoord[0], pixelcoord[1], marker='o', markersize=5, color='white')
        ax.annotate(hess_catalog['Source_Name'][i],xy=(pixelcoord[0], pixelcoord[1]), xycoords='data', xytext=(100, 145), 
        textcoords='offset points', arrowprops=dict(arrowstyle="-",color='white', linewidth=2, 
        linestyle='-.'), color='gray', rotation=0, ha='right', va='center' ,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        if hess_catalog['Size'][i] != ' ':
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), float(hess_catalog['Size'][i])/0.005, color='white', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)
        else:
            hawcext = plt.Circle((pixelcoord[0], pixelcoord[1]), float(0.01)/0.005, color='white', linewidth=2, fill=False, linestyle='-')
            ax.add_patch(hawcext)


def parse_model_file(catalog_results):
    with open(catalog_results, "r") as f:
        content = f.read()

    source_blocks = re.findall(r"source_name\s*=\s*\"(.*?)\"(.*?)###################################", content, re.DOTALL)
    sources = []

    for name, block in source_blocks:
        ra_match = re.search(r"source_ra\s*=\s*([\d.]+)", block)
        dec_match = re.search(r"source_dec\s*=\s*([\d.]+)", block)
        sigma_match = re.search(r"shape\s*=\s*astromodels\.Gaussian_on_sphere\(\).*?shape\.sigma\s*=\s*([\d.]+)", block, re.DOTALL)
        source_info = {
            "name": name,
            "ra": float(ra_match.group(1)) if ra_match else None,
            "dec": float(dec_match.group(1)) if dec_match else None,
            "sigma": float(sigma_match.group(1)) if sigma_match else 0.0,
        }

        sources.append(source_info)
    return sources