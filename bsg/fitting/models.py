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


def symmetric_gaussian_func(nu, amplitude, mean, std):
    return amplitude * np.exp(-0.5 * ((nu - mean) / std) ** 2)


def harvey_func(frequencies, a, b):
    return (a/b) / (1 + (frequencies / b) ** 2) 


def white_noise_func(frequencies, white_noise):
    return np.ones_like(frequencies) * white_noise


def fit_func(nu, gaussian_amplitude, gaussian_mean, gaussian_std, 
                 harvey_amplitude, harvey_timescale, 
                 white_noise):

    return symmetric_gaussian_func(nu, gaussian_amplitude, gaussian_mean, gaussian_std) + \
                                        harvey_func(nu, harvey_amplitude, harvey_timescale) + \
                                        white_noise_func(nu, white_noise)

    return fitted_parameters


def run_fit(nu, amp):

    # Initial guess for the parameters
    # initial_guess = [gaussian_amplitude, gaussian_mean, gaussian_std, harvey_amplitude, harvey_timescale, white_noise]
    initial_guess = [np.max(amp), nu[np.argmax(amp)], 0.1, 0.01*np.max(amp), 0.2, 0.001*np.max(amp)]


    # Perform the curve fitting
    fitted_parameters, _ = curve_fit(fit_func, nu, amp, p0=initial_guess)

    return fitted_parameters



def fit_excess(tic, times, fluxes, max_frequency=25):

    # Compute Lomb-Scargle periodogram
    # Don't forget to remove the median of the flux array
    nu, amp = LS_periodogram(times, fluxes-np.median(flux), max=max_frequency)


    # Smooth the amplitude array
    window_size = len(nu[nu < 0.2])
    amp_smoothed = smooth(amp, window_size, 'gaussian')


    # Fit a skewed Gaussian on top of a Harvey profile with white noise offset
    # Your code for fitting the skewed Gaussian goes here
    fitted_parameters = run_fit(nu, amp_smoothed)

    fig, ax = plt.subplots(1,1,figsize=(9, 6))
    ax.loglog(nu, amp*1e6, label='Original')
    ax.loglog(nu, amp_smoothed*1e6, label='Smoothed')
    ax.loglog(nu, fit_func(nu, *fitted_parameters)*1e6, 'k-', label='Fitted')
    ax.loglog(nu, symmetric_gaussian_func(nu, *fitted_parameters[:3])*1e6, ':', label='Gaussian')
    ax.loglog(nu, harvey_func(nu, *fitted_parameters[3:5])*1e6, ':', label='Harvey')    
    ax.loglog(nu, white_noise_func(nu, fitted_parameters[5])*1e6, ':', label='White noise')
    ax.set_ylim(1e-1, 1e4)
    ax.set_xlim(1e-3, 25)
    ax.set_xlabel(r'${\rm Frequency~[d^{-1}]}$')
    ax.set_ylabel(r'${\rm Amplitude~[ppm]}$')
    ax.legend()
    fig.suptitle(r'${\rm TIC~}$'+f'{tic}')
    fig.tight_layout()
    fig.savefig('./lmc_pngs/TIC{}_fit.png'.format(tic))
    # fig.savefig('./gal_pngs/TIC{}_fit.png'.format(tic))
    fig.clf()
    plt.close()
    # plt.show()


    # Return the fitted parameters
    return fitted_parameters


if __name__ == '__main__':

    # LMC
    files = glob('./lmc_data/TIC*.dat')

    dc = {'TIC': [], 'gaussian_amplitudes': [], 'gaussian_means': [], 
            'gaussian_widths': [], 'harvey_amplitudes': [], 
            'harvey_timescales': [], 'white_noises': []}

    for file in files:
        tic = file.split('TIC')[1].split('_')[0]
        times, flux, ferr = np.loadtxt(file, unpack=True)

        fitted_parameters = fit_excess(tic, times, flux)

        dc['TIC'].append(tic)
        dc['gaussian_amplitudes'].append(fitted_parameters[0])
        dc['gaussian_means'].append(fitted_parameters[1])
        dc['gaussian_widths'].append(fitted_parameters[2])
        dc['harvey_amplitudes'].append(fitted_parameters[3])
        dc['harvey_timescales'].append(fitted_parameters[4])
        dc['white_noises'].append(fitted_parameters[5])

    df = pd.DataFrame(dc)
    df.to_csv('fitted_parameters.csv', index=False)

    fig, axes = plt.subplots(2, 3, figsize=(9, 9))
    axes[0][0].plot(df['gaussian_means'],df['gaussian_amplitudes']*1e6, 'ko', label='Gaussian Amplitude')
    axes[0][1].plot(df['gaussian_means'],df['gaussian_widths'], 'ro', label='Gaussian Width')
    axes[0][2].plot(df['gaussian_means'],df['harvey_amplitudes']*1e6, 'bo', label='Harvey Amplitude')
    axes[1][0].plot(df['gaussian_means'],df['harvey_timescales'], 'go', label='Harvey Timescale')
    axes[1][1].plot(df['gaussian_means'],df['white_noises']*1e6, 'mo', label='White Noise')
    for i in range(2):
        for j in range(3):
            axes[i][j].set_xlabel(r'${\rm Frequency [d^{-1}]}$')
    axes[0][0].set_ylabel(r'${\rm Amplitude~[ppm]}$')
    axes[0][1].set_ylabel(r'${\rm Width~[d^{-1}]}$')
    axes[0][2].set_ylabel(r'${\rm Amplitude~[ppm]}$')
    axes[1][0].set_ylabel(r'${\rm Timescale~[d]}$')
    axes[1][1].set_ylabel(r'${\rm White~Noise~[ppm]}$')
    fig.tight_layout()
    fig.savefig('fitted_parameters.png')
    plt.show()
