import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from pythia.timeseries.lombscargle import LS_periodogram
from scipy.optimize import curve_fit
from .samplers import run_model
from bsg.utils.processing import smooth
from bsg.fitting.terms import skewed_gaussian_func, symmetric_gaussian_func, harvey_func, white_noise_func

def fit_func(x, gaussian_amplitude, gaussian_mean, gaussian_std, 
                harvey_amplitude, harvey_timescale,
                white_noise):

    return symmetric_gaussian_func(x, gaussian_amplitude, gaussian_mean, gaussian_std) + \
           harvey_func(x, harvey_amplitude, harvey_timescale) + \
           white_noise_func(x, white_noise)


def run_optimizer(x,y,conversion):

    # Initial guess for the parameters
    # initial_guess = [gaussian_amplitude, gaussian_mean, gaussian_std, harvey_amplitude, harvey_timescale, white_noise]
    initial_guess = [np.max(y), x[np.argmax(y)], 0.1*conversion, 0.005*np.max(y), 0.2*conversion, np.median(y[-100:])]

    # Perform the curve fitting
    fit_parameters, _ = curve_fit(fit_func, x, y, p0=initial_guess)

    fit_dict = {'gaussian_amplitude': fit_parameters[0], 'gaussian_mean': fit_parameters[1], 'gaussian_std': fit_parameters[2],
                'harvey_amplitude': fit_parameters[3], 'harvey_timescale': fit_parameters[4],
                'white_noise': fit_parameters[5]}
    return fit_dict


def run_sampler(x,y, colored_noise='Harvey'):

    trace = run_model(x, y, colored_noise=colored_noise)
        
    return trace

def fit_model( tic, times, fluxes, max_frequency=25, normalization='amplitude', 
               optimizer=True, optimizer_kwargs=None,
               sampler=False, sampler_kwargs=None ):

    # Compute Lomb-Scargle periodogram
    # Don't forget to remove the median of the flux array
    nu, amp = LS_periodogram(times, fluxes-np.median(fluxes), max=max_frequency)
    amp*=1e6

    if normalization == 'amplitude':
        x = nu
        y = amp
        conversion = 1.

    elif normalization == 'psd':

        conversion = 1.e6 / 86400.
        x = nu*conversion
        resolution = conversion / (times[-1] - times[0])
        y = (amp**2) / resolution

    # Smooth the amplitude array
    window_size = len(x[x < (0.2 * conversion)])
    y_smoothed = np.array(smooth(y, window_size, 'gaussian'))
    print(np.shape(x),np.shape(y_smoothed))

    # plt.loglog(x, y, 'k-')
    # plt.loglog(x, y_smoothed, '-', color='darkorange')
    # plt.show()

    if sampler:
        # Fit a skewed Gaussian on top of a Harvey profile with white noise offset
        # Your code for fitting the skewed Gaussian goes here
        output = run_sampler(x, y_smoothed)
        # output = pm.summary(trace)

    elif optimizer:
        # Fit a skewed Gaussian on top of a Harvey profile with white noise offset
        # Your code for fitting the skewed Gaussian goes here
        fit_parameters = run_optimizer(x, y_smoothed, conversion)
        output = fit_parameters
    else:
        print('No optimizer or sampler selected. \n Please choose one of the two.')


    # Return the fitted parameters
    return x, y, y_smoothed, output
