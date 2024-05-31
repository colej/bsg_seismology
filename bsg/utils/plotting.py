## Taken from noraeisner/LATTE
## modified for our purposes
## Please cite:
# @ARTICLE{2020JOSS....5.2101E,
#        author = {{Eisner}, Nora and {Lintott}, Chris and {Aigrain}, Suzanne},
#         title = "{LATTE: Lightcurve Analysis Tool for Transiting Exoplanets}",
#       journal = {The Journal of Open Source Software},
#      keywords = {exoplanets, Python, transit, Jupyter Notebook, astronomy, TESS},
#          year = 2020,
#         month = may,
#        volume = {5},
#        number = {49},
#           eid = {2101},
#         pages = {2101},
#           doi = {10.21105/joss.02101},
#        adsurl = {https://ui.adsabs.harvard.edu/abs/2020JOSS....5.2101E},
#       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
# }

import os
import csv
import sys
import json
import astropy
import requests
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sb
import lightkurve as lk
from os.path import exists
import matplotlib.pyplot as plt

from astropy.wcs import WCS
import astropy.io.fits as pf
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord


from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs

from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons

import http.client as httplib
import mpl_toolkits.axes_grid1
from astroquery.mast import Catalogs
from sklearn.decomposition import PCA
from urllib.parse import quote as urlencode
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astroplan import FixedTarget
from astroplan.plots import plot_finder_image

from bsg.fitting import models


main_plot_color = '#ffa31a'
main_plot_fontsize = 13
md_color = '#143085'
cutout_window = 0.7
individual_ylim = False


# plot the nearby TESS stars as well as the SDSS cutout - reprojected to have North up.
def plot_TESS_stars(starName, indir, tpf, fig, gs1, gs2, show=False, save=True):

    '''
    Plot of the field of view round the target star showing nearby stars that are brighter than magnitude 17 as well as the SDSS cutout.
    Both images are projected and oriented North for easy comparison.

    Parameters
    ----------
    tic : str
        TIC (Tess Input Catalog) ID of the target
    indir   :   str
        path to where the data will be saved (defaul = "./LATTE_output")
    tpf   : hdu object
        target pixel file

    Returns
    -------
        Plot of the averaged flux per pixel around the target (left) as well as the SDSS plot (right). The red star on the right plot indicated the location of the target.
        The orange circles show the location of nearby stars with magnitudes brighter than 17 mag where their relative sizes correspond to their relative brightness.
        The location of the target star is hown with the reticle on the right hans side SDSS image.

    Tmag   :   float
        TESS magnitude of the target star
    Teff   :   float
        Effective temperature of the target star (K)
    rad   :   float
        radius of the target star (Solar radii)
    mass   :   float
        mass of the target star (Solar masses)
    '''

    # ----------
    # import the astroplan module that is needed - this is done here and not at the start at this scipt
    # because astroplan cannot be parallelised (issues with python's shelve storage) so import here.
    # ----------

    # Query nearby Gaia Stars  --------------------

    radSearch = 5/60 #radius in degrees

    # this function depends on astroquery working, and sometimes it doesn't.
    # for when it doesn't work (or simply can't connect to it), just skip plotting the other TESS stars.
    try:
        catalogData = Catalogs.query_object(starName, radius = radSearch, catalog = "TIC")
    except:
        print ("Currently cannot connect to Astroquery.")
        # return values that we know aren't real so that we can tell the code that the plotting didn't work
        return -999, -999, -999, 1, -999,-999,-999,-999

    # ra and dec of the target star
    ra = catalogData[0]['ra']
    dec = catalogData[0]['dec']

    # while we have the astroquery loaded, let's collect some other information about the star
    # these paramaters can help us find out what type of star we have with just a glance

    vmag = catalogData['Vmag'][0] # v magnitude (this migth be more useful than the TESS mag for things such as osbevring)
    logg = catalogData['logg'][0] # logg of the star
    mass = catalogData['mass'][0] # mass of the star
    plx = catalogData['plx'][0]   # parallax

    # sometimes these values aren't accessible through astroquery - so we shoudl just quickly check.
    if not np.isfinite(vmag): vmag = '--' # this is what will appear in the table of the report to indicate that it's unknown
    if not np.isfinite(logg): logg = '--'
    if not np.isfinite(mass): mass = '--'
    if not np.isfinite(plx): plx   = '--'

    # sometimes it's useufl to know if the star has another name
    # check whether it was osberved by one of these four large surveys

    catalogs = ['HIP', 'TYC', 'TWOMASS', 'GAIA']

    for cat in catalogs:
        c_id = str(catalogData[0][cat])
        if c_id != '--':
            c_id = "{} {}".format(cat,c_id)
            break
        else:
            continue

    # ------------------------------------------

    # Create a list of nearby bright stars (tess magnitude less than 17) from the rest of the data for later.
    bright = catalogData['Tmag'] < catalogData['Tmag'][0] + 4

    # ---------------------------------------------
    # Get the data for the SDSS sky viewer --------

    survey = 'DSS2 Red'
    fig_tmp, ax_tmp = plt.subplots()
    # fig_tmp.axis("off")

    target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    try:
        target = FixedTarget(coord=target_coord, name="Survey = {}".format(survey))
        ax_tmp, hdu = plot_finder_image(target, survey = survey, ax=ax_tmp, reticle='True', fov_radius=5*u.arcmin)

    except: # if DSS2 Red is not available, download the DSS field of view image instead
        try:
            survey = 'DSS'
            target = FixedTarget(coord=target_coord, name="Survey = {}".format(survey))
            ax_tmp, hdu = plot_finder_image(target, survey = survey, ax=ax_tmp, reticle='True', fov_radius=5*u.arcmin)
        except:
            return -111,-111,-111,-111,-111,-111,-111,-111

    plt.close(fig_tmp)

    # --------------------------------------------

    sector = tpf.header['SECTOR']

    tpf_wcs = tpf.wcs
    tup = (np.nanmean(tpf.flux, axis=0),tpf_wcs)
    # create a tupple of the array of the data and the wcs projection of the TESS cutout


    # map the SDSS and TESS image onto each other - the output will be orented NORTH!
    wcs_out, shape_out = find_optimal_celestial_wcs(input_data =[tup, hdu])


    # plot the reprojected TESS image
    ax_tpf = fig.add_subplot(gs1, projection=wcs_out)
    array, footprint = reproject_interp(tup, wcs_out,shape_out = shape_out,order = 'nearest-neighbor')

    ax_tpf.imshow(array, origin='lower', cmap = plt.cm.YlGnBu_r)

    ax_tpf.coords['ra'].set_axislabel('Right Ascension', fontsize = main_plot_fontsize)
    ax_tpf.coords['dec'].set_axislabel('Declination', fontsize = main_plot_fontsize)
    ax_tpf.grid(color = 'grey', alpha = 0.7)

    # plot the nearby GAIA stars on this image too...
    ra_stars, dec_stars = catalogData[bright]['ra'], catalogData[bright]['dec']
    s = np.maximum((19 - catalogData[bright]['Tmag'])*5, 0)  # the size corresponds to their brightness
    ax_tpf.scatter(ra_stars, dec_stars, s=s, transform=ax_tpf.get_transform('icrs'), color=main_plot_color, zorder=100)

    # plot the target star that we're looking at
    ax_tpf.scatter(ra, dec, s= 200, transform=ax_tpf.get_transform('icrs'), marker = '*', color='red', zorder=100)
    ax_tpf.tick_params(labelsize=main_plot_fontsize)

    # plot the reprojected SDSS image
    ax_dss = fig.add_subplot(gs2, projection=wcs_out, sharex=ax_tpf, sharey=ax_tpf)
    array, footprint = reproject_interp(tup, wcs_out,shape_out = shape_out)
    ax_dss.imshow(hdu.data, origin='lower', cmap = 'Greys')
    ax_dss.coords['ra'].set_axislabel('Right Ascension', fontsize = main_plot_fontsize)
    #ax_dss.coords['dec'].set_axislabel('Declination')

    # Draw reticle ontop of the target star
    pixel_width = hdu.data.shape[0]
    inner, outer = 0.03, 0.08

    reticle_style_kwargs = {}
    reticle_style_kwargs.setdefault('linewidth', 1.5)
    reticle_style_kwargs.setdefault('color', 'red')

    ax_dss.axvline(x=0.5*pixel_width, ymin=0.5+inner, ymax=0.5+outer,
               **reticle_style_kwargs)
    ax_dss.axvline(x=0.5*pixel_width, ymin=0.5-inner, ymax=0.5-outer,
               **reticle_style_kwargs)
    ax_dss.axhline(y=0.5*pixel_width, xmin=0.5+inner, xmax=0.5+outer,
               **reticle_style_kwargs)
    ax_dss.axhline(y=0.5*pixel_width, xmin=0.5-inner, xmax=0.5-outer,
                   **reticle_style_kwargs)
    ax_dss.grid()
    ax_dss.tick_params(labelsize=main_plot_fontsize)
    plt.tight_layout(w_pad = 5)

    if save:
        plt.savefig('{}tic{}_star_field.png'.format(indir, starName.split(' ')[-1]), format='png', bbox_inches='tight')

    # if show:
    #     plt.show()
    # else:
    #     plt.close()

    return fig, catalogData['Tmag'][0], catalogData['Teff'][0], catalogData['rad'][0], mass, vmag, logg, plx, c_id


# LC per pixel
def plot_pixel_level_LC(tic, indir, X1_list, X4_list, oot_list, intr_list, bkg_list, tpf_list, apmask_list, arrshape_list, t_list, transit_list, args, ql = False):

    '''
    Plot the LC for each pixel around the time of the transit like event. Each LC is fit with a spline and corrected to flatten.
    each LC is fitted with a 3 order polynomial in order to flatten.

    Parameters
    ----------
    tic : str
        TIC (Tess Input Catalog) ID of the target
    indir   :   str
        path to where the data will be saved (defaul = "./LATTE_output")
    X1_list  :  list
        flux vs time for each pixel
    X4_list  :  list
        PCA corrected flux vs time for each pixel
    oot_list  :  list
        out of transit mask
    intr_list  :  list
        in transit mask
    bkg_list  :  list
        the flux that was used to normalise each pixel - i.e. what is used to make the background plot colour for each pixel.
    apmask_list  :  list
        aperture masks from the pipeline
    arrshape_list  :  list
        shape of the array
    t_list  :  list
        time arrays
    transit_list  :  int
        list of all the marked transits

    Returns
    -------
        Plot of the normalised LC for each pixel around the time of the transit like event.
        The pixel backrgound colour represents the average flux.
        The time of the transit is highlighted in red/gold for each pixel LC.
    '''

    # loop through the transits and make plot for each ( only the first is currently displayed in the pdf report)
    for idx, X1 in enumerate(X1_list):


        X4 = X4_list[idx]
        oot = oot_list[idx]
        #intr = intr_list[n]
        bkg = np.flip(bkg_list[idx], axis = 0)
        arrshape = arrshape_list[idx]
        t = t_list[idx]
        peak = transit_list[idx]
        tpf = tpf_list[idx]

        if args.FFI != 'QLP':
            apmask = apmask_list[idx]

            mapimg = np.flip(apmask_list[idx], axis = 0)

            ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
            hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

        fig, ax = plt.subplots(arrshape[1], arrshape[2], sharex = True, sharey = False, gridspec_kw={'hspace': 0 ,'wspace': 0}, figsize=(8,8))

        plt.tight_layout()

        # see if the backrgound of this plot can be the average pixel flux (if there are too many nans this will fail and the background will just be black which is also okay)
        try:
            color = plt.cm.viridis(np.linspace(0, 1,int(np.nanmax(bkg))-int(np.nanmin(bkg))+1))
            simplebkg = False
        except:
            simplebkg = True

        for i in range(0,arrshape[1]):
            print ("{}   out of    {} ".format(i+1,arrshape[1] ))
            ii = arrshape[1]-1-i # we want to plot this such that the pixels increase from left to right and bottom to top

            for j in range(0,arrshape[2]):

                apmask = np.zeros(arrshape[1:], dtype=np.int)
                apmask[i,j] = 1
                apmask = apmask.astype(bool)

                flux = X1[:,apmask.flatten()].sum(axis=1)

                m = np.nanmedian(flux[oot])

                normalizedflux = flux/m

                # bin the data
                f1 = normalizedflux
                time = t

                if args.FFI == False:
                    binfac = 5

                    N       = len(time)
                    n       = int(np.floor(N/binfac)*binfac)
                    X       = np.zeros((2,n))
                    X[0,:]  = time[:n]
                    X[1,:]  = f1[:n]
                    Xb      = rebin(X, (2,int(n/binfac)))

                    # binned data
                    time_binned    =    np.array(Xb[0])
                    flux_binned    =   np.array(Xb[1])

                else:
                    # binned data -
                    time_binned    =    np.array(time)
                    flux_binned  =   np.array(flux)

                # create a mask that only looks at the times cut around the transit-event
                timemask = (time_binned < peak+1.5) & (time_binned > peak-1.5)

                time_binned = time_binned[timemask]
                flux_binned = flux_binned[timemask]

                # ----------
                # fit a spline to the cut-out of each pixel LC in order to flatten it
                p = np.poly1d(np.polyfit(time_binned, flux_binned, 3))
                flux_binned = flux_binned/p(time_binned)
                # ----------

                intr = abs(peak-time_binned) < 0.1

                if simplebkg == True:
                    ax[ii, j].set_facecolor(color = 'k')
                    linecolor = 'w'
                    transitcolor = 'gold'
                else:
                    ax[ii, j].set_facecolor(color = color[int(bkg[ii,j])-int(np.nanmin(bkg))])

                    if int(bkg[ii,j])-abs(int(np.nanmin(bkg))) > ((np.nanmax(bkg))-abs(int(np.nanmin(bkg))))/2:
                        linecolor = 'k'
                        transitcolor = 'orangered'
                    else:
                        linecolor = 'w'
                        transitcolor = 'gold'


                ax[ii, j].plot(time_binned,flux_binned, color = linecolor, marker = '.', markersize=1, lw = 0)
                ax[ii, j].plot(time_binned[intr],flux_binned[intr], color = transitcolor, marker = '.', markersize=1, lw = 0)

                # get rid of ticks and ticklabels
                ax[ii,j].set_yticklabels([])
                ax[ii,j].set_xticklabels([])
                ax[ii,j].set_xticks([])
                ax[ii,j].set_yticks([])

        # ------------------
        if args.FFI != 'QLP':
            print ("\n Calculating the Aperture Mask...", end =" ")

            for i in range(0,len(ver_seg[1])):
                ax[ver_seg[0][i], ver_seg[1][i]].spines['right'].set_color('red')
                ax[ver_seg[0][i], ver_seg[1][i]].spines['right'].set_linewidth(6)

                top = (ax[ver_seg[0][i], ver_seg[1][i]].get_ylim()[1])
                bottom= (ax[ver_seg[0][i], ver_seg[1][i]].get_ylim()[0])

                change = top - bottom

                ax[ver_seg[0][i], ver_seg[1][i]].spines['right'].set_bounds(bottom,top - (change*0.08) )

                for j in range(0,len(hor_seg[1])):
                    ax[hor_seg[0][j], hor_seg[1][j]].spines['bottom'].set_color('red')
                    ax[hor_seg[0][j], hor_seg[1][j]].spines['bottom'].set_linewidth(6)
                    ax[hor_seg[0][j], hor_seg[1][j]].spines['bottom'].set_bounds(peak-1.5,peak+1.3)

        print ("done.\n")
        # ------------------

        # label the pixels
        if (args.FFI == False) or (args.FFI == 'SPOC'):
            start_column = tpf[1].header['1CRV5P']
            start_row = tpf[1].header['2CRV5P']

        else:
            start_column = tpf.get_keyword('1CRV5P', hdu=1, default=0)
            start_row = tpf.get_keyword('2CRV5P', hdu=1, default=0)

        y_start = '{}'.format(start_row)
        y_end = '{}'.format(start_row + bkg.shape[0])

        x_start = '{}'.format(start_column)
        x_end = '{}'.format(start_column + bkg.shape[1])

        ax[bkg.shape[0]-1, 0].set_xlabel(x_start)
        ax[bkg.shape[0]-1, 0].set_ylabel(y_start)

        ax[0,0].set_ylabel(y_end)
        ax[bkg.shape[0]-1,bkg.shape[1]-1].set_xlabel(x_end)

        fig.text(0.5,0.01, "column (pixel)", ha='center', fontsize = 13)
        fig.text(0.01, 0.5, "row (pixel)", va='center', rotation='vertical', fontsize = 13)

        # - - - - - - - - - -

        plt.subplots_adjust(top=0.95, right = 0.99, bottom = 0.04, left = 0.04)


        plt.suptitle(r"T0 = {} $\pm$ 1.5 d".format(peak ),y=0.98, fontsize = 15)
        plt.xlim(peak-1.5,peak+1.5)

        if (args.save == True) and (ql == False):
            print ("Waiting on plot...")
            plt.savefig('{}/{}/{}_individual_pixel_LCs_{}.png'.format(indir, tic,tic, idx), format='png')

        if (args.noshow == False) or (ql == True):
            plt.show()
        else:
            plt.close()



def extract_wcs(tpf):
    '''
    Extract the wcs from the header - function from lighkurve
    '''
    wcs_keywords = {'1CTYP5': 'CTYPE1',
                    '2CTYP5': 'CTYPE2',
                    '1CRPX5': 'CRPIX1',
                    '2CRPX5': 'CRPIX2',
                    '1CRVL5': 'CRVAL1',
                    '2CRVL5': 'CRVAL2',
                    '1CUNI5': 'CUNIT1',
                    '2CUNI5': 'CUNIT2',
                    '1CDLT5': 'CDELT1',
                    '2CDLT5': 'CDELT2',
                    '11PC5': 'PC1_1',
                    '12PC5': 'PC1_2',
                    '21PC5': 'PC2_1',
                    '22PC5': 'PC2_2',
                    'NAXIS1': 'NAXIS1',
                    'NAXIS2': 'NAXIS2'}
    mywcs = {}

    for oldkey, newkey in wcs_keywords.items():
        if (tpf[1].header[oldkey] != pf.Undefined):
            mywcs[newkey] = tpf[1].header[oldkey]
    return WCS(mywcs)


def plot_periodogram_fit(x, y, y_smoothed, fit_parameters, 
                         savedir=None, tic=None, 
                         xlabel=r'${\rm Frequency~[d^{-1}]}$',
                         ylabel=r'${\rm Amplitude~[ppm]}$'):

    arr = np.array([fit_parameters[par] for par in fit_parameters])
    fig, ax = plt.subplots(1,1,figsize=(9, 6))
    ax.loglog(x, y, label='Original')
    ax.loglog(x, y_smoothed, label='Smoothed')
    ax.loglog(x, models.fit_func(x, *arr), 'k-', label='Fitted')
    ax.loglog(x, models.symmetric_gaussian_func(x, *arr[:3]), ':', label='Gaussian')
    ax.loglog(x, models.harvey_func(x, *arr[3:5]), ':', label='Harvey')    
    ax.loglog(x, models.white_noise_func(x, arr[-1]), ':', label='White noise')
    ax.set_ylim(1e-1, max(y)*1.2)
    ax.set_xlim(min(x), max(x)+5.)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.suptitle(r'${\rm TIC~}$'+f'{tic}')
    fig.tight_layout()
    if savedir is not None:
        fig.savefig('{}/TIC{}_fit.png'.format(savedir,tic))

    return fig, ax
