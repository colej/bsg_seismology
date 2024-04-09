import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from pythia.timeseries.lombscargle import LS_periodogram
from scipy.optimize import curve_fit


def gaussian04(N):
    n = np.arange(0,N)
    return np.exp(-0.5*((n-(N-1)/2)/(0.4*(N+1)/2))**2.)


def smooth(x,window_len,window):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,num=1000)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    !! Credit to @BRAM BUYSSCHAERT for original code
    """

    if x.ndim != 1:
        raise ValueError("Only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        raise ValueError("Window must be at least 3 points.")

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','gaussian']:
        raise ValueError("Window must be either 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','gaussian'")

    window_len = int(window_len)

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        if window == 'gaussian':
            w=gaussian04(window_len)
        else:
            w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    if window_len%2 == 0:
      return y[int(window_len/2-1):-int(window_len/2)] #NOTE
    else:
      return y[int(window_len/2):-int(window_len/2)]


def skewed_gaussian_func(x, a, b, c, d, e):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2) * (1 + erf((x - b) / (d * np.sqrt(2)))) + e


def symmetric_gaussian_func(x, amplitude, mean, std):
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)


def harvey_func(x, a, b):
    return (a/b) / (1 + (x / b) ** 2) 


def white_noise_func(x, white_noise):
    return np.ones_like(x) * white_noise


def fit_func(x, gaussian_amplitude, gaussian_mean, gaussian_std, 
                 harvey_amplitude, harvey_timescale, 
                 white_noise):

    return symmetric_gaussian_func(x, gaussian_amplitude, gaussian_mean, gaussian_std) + \
                                        harvey_func(x, harvey_amplitude, harvey_timescale) + \
                                        white_noise_func(x, white_noise)


def run_fit(x,y):

    # Initial guess for the parameters
    # initial_guess = [gaussian_amplitude, gaussian_mean, gaussian_std, harvey_amplitude, harvey_timescale, white_noise]
    initial_guess = [np.max(y), x[np.argmax(y)], 0.1, 0.01*np.max(y), 0.2, 0.001*np.max(y)]


    # Perform the curve fitting
    fit_parameters, _ = curve_fit(fit_func, x, y, p0=initial_guess)

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
        y = (amp**2) / x

    # Smooth the amplitude array
    window_size = len(x[x < (0.2 * conversion)])
    y_smoothed = smooth(y, window_size, 'gaussian')


    # Fit a skewed Gaussian on top of a Harvey profile with white noise offset
    # Your code for fitting the skewed Gaussian goes here
    fit_parameters = run_fit(x, y_smoothed)


    # Return the fitted parameters
    return x, y, y_smoothed, fit_parameters
