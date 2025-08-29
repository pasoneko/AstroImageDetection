# DRIPS Image Processing Source Search

## Getting started

Instructions:
1. Clone the repository
2. Enable your HAL environment
3. Run `finder.py` with the arguments specified below

## Arguments for `finder.py`:
### Required arguments:
- [ ] `-M`, `--map` : "Significance Map file in root format" 
- [ ] `--ROI-center` : ROI Center of the image (ra, dec) or (l, b)
- [ ] `--coordsys` : Image Coordinate: 'G', 'C'.  (Default: 'G') 
- [ ] `--size` : ROI Size for the image(Default: 5 x 5 degrees)
### Optional arguments:
- [ ] `-O`, `--outdir` : output directory for any generated plots (recommended, otherwise old plots will be overwritten)
- [ ] `--plot4HWC`, `--plotlhaaso`, `--plotHGPS`, `--plotFermi`(WiP), `--plotSNR`, `--plotPulsar`(WiP) are for plotting respective catalogs, example usage `--plot4HWC True`
- [ ] `--plotPDF` : save all plots in a PDF
- [ ] `--saveModel` : saveModel for final fits
- [ ] `--plotSurface` : plot the surface intensity peak maps
- [ ] `--plotStackSignif` : Plot Change in Significance Ranges

```
NOTE: 
- [ ] Works with default HAL. Might need to install scikit-image package
- [ ] For more details refer to:
- [1] https://private.hawc-observatory.org/wiki/images/8/86/Hawc_im_process.pdf
- [2] https://private.hawc-observatory.org/wiki/images/0/02/DA21January2025_Rishi.pdf

```

If you there are comments/questions/suggestion on the running the script, contact Rishi Babu (rbabu@icecube.wisc.edu) or Palmer Wentworth (itomura@msu.edu).

