import numpy as np

def skewed_gaussian_func(x, a, b, c, d, e):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2) * (1 + erf((x - b) / (d * np.sqrt(2)))) + e


def symmetric_gaussian_func(x, amplitude, mean, std):
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)


def harvey_func(x, a, b):
    return (2.*np.pi*a/b) / (1 + (x / b) ** 2)


def white_noise_func(x, white_noise):
    return np.ones_like(x) * white_noise


def ma_func(x, a, b, gamma):
    return (a * x/b) / (1 + (x / b) ** gamma)