import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from pythia.timeseries.lombscargle import LS_periodogram
from scipy.optimize import curve_fit






def fit_func(x, gaussian_amplitude, gaussian_mean, gaussian_std, 
                 harvey_amplitude, harvey_timescale,
                 white_noise):

    return symmetric_gaussian_func(x, gaussian_amplitude, gaussian_mean, gaussian_std) + \
                                        harvey_func(x, harvey_amplitude, harvey_timescale) + \
                                        white_noise_func(x, white_noise)


def run_fit(x,y,conversion):

    # Initial guess for the parameters
    # initial_guess = [gaussian_amplitude, gaussian_mean, gaussian_std, harvey_amplitude, harvey_timescale, white_noise]
    initial_guess = [np.max(y), x[np.argmax(y)], 0.1*conversion, 0.005*np.max(y), 0.2*conversion, np.median(y[-100:])]

    # Perform the curve fitting
    fit_parameters, _ = curve_fit(fit_func, x, y, p0=initial_guess)

    print(fit_parameters[-1])
    return fit_parameters



def fit_excess(tic, times, fluxes, max_frequency=25, normalization='amplitude'):

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
    y_smoothed = smooth(y, window_size, 'gaussian')


    # Fit a skewed Gaussian on top of a Harvey profile with white noise offset
    # Your code for fitting the skewed Gaussian goes here
    fit_parameters = run_fit(x, y_smoothed, conversion)


    # Return the fitted parameters
    return x, y, y_smoothed, fit_parameters
