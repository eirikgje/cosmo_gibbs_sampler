#Tools for analysing the results from the sampler
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def calculate_ml_and_asymm_errorbars_from_samples(samples, x=None, area_fraction=0.68, smooth=True):
    #Assume samples is a fits file. Pretty simple now.
    if smooth:
        kernel = scipy.stats.kde.gaussian_kde(samples)
        ml_ind = np.argmax(kernel(x))
        ml = x[ml_ind]
        dx = x[1] - x[0]
        currpoint_high = ml_ind
        currpoint_low = ml_ind
        y = kernel(x)
        if y[ml_ind + 1] > y[ml_ind - 1]:
            currpoint_high += 1
        else:
            currpoint_low -= 1
        area = sum(y[currpoint_low:currpoint_high + 1]*dx)
        while area < area_fraction:
            if (currpoint_high == len(x) - 1 and currpoint_low == 0):
                raise ValueError("Cannot find bounds")
            if currpoint_high == len(x) - 1:
                currpoint_low -= 1
            elif currpoint_low == 0:
                currpoint_high += 1
            else:
                if y[currpoint_high + 1] > y[currpoint_low - 1]:
                    currpoint_high += 1
                else:
                    currpoint_low -= 1
            area = sum(y[currpoint_low:currpoint_high + 1] * dx)
    
        upper = x[currpoint_high]
        lower = x[currpoint_low]
    else:
        sortedSamps = np.sort(samples)
        if x is None:
            x = np.linspace(sortedSamps[0], sortedSamps[-1], 10000)
        hist = scipy.stats.histogram(sortedSamps, numbins=len(x), defaultlimits=(x[0], x[-1]))
        ml = x[np.argmax(hist[0])]
        mlarg = np.searchsorted(sortedSamps, np.array(ml))
        numsamp_thresh = int(round(area_fraction * len(samples)))
        if mlarg == 0:
            lower = sortedSamps[0]
            upper = sortedSamps[numsamp_thresh]
            return ml, upper, lower
        if mlarg < numsamp_thresh:
            currminbracket = sortedSamps[numsamp_thresh] - sortedSamps[0]
            currminind = 0
            start = 1
            if mlarg + numsamp_thresh >= len(samples):
                stop = len(samples) - numsamp_thresh
            else:
                stop = mlarg + 1
        elif mlarg + numsamp_thresh < len(samples):
            currminbracket = sortedSamps[mlarg] - sortedSamps[mlarg - numsamp_thresh]
            currminind = mlarg - numsamp_thresh
            start = mlarg - numsamp_thresh + 1
            stop = mlarg + 1
        elif mlarg + numsamp_thresh >= len(samples):
            currminbracket = sortedSamps[mlarg] - sortedSamps[mlarg - numsamp_thresh]
            currminind = mlarg - numsamp_thresh
            start = mlarg - numsamp_thresh + 1
            stop = len(samples) - numsamp_thresh
        for i in range(start, stop):
            currbrack = sortedSamps[numsamp_thresh + i] - sortedSamps[i]
            if currbrack < currminbracket:
                currminbracket = currbrack
                currminind = i
        upper = sortedSamps[currminind + numsamp_thresh]
        lower = sortedSamps[currminind]

    return (ml, upper, lower)

def plot_ml_powspec_with_band(sigmas, lmax=100, sample_fraction=0.68, label=None, color='red', mlcolor='blue',spec=[0,0], burnin=0):

    ls = np.arange(2, lmax + 1)
    res = np.zeros((len(ls), 3))
    i = 0
    for l in ls:
        samps = sigmas[:, l, spec[0], spec[1]] * l * (l+1) / (2 * np.pi)
        x = np.linspace(min(samps), max(samps), 100)
        ml, upper, lower = calculate_ml_and_asymm_errorbars_from_samples(samps, x, sample_fraction, smooth=False)
        res[i, 0] = ml
#        res[i, 1] = ml - lower
#        res[i, 2] = upper - ml
        res[i, 1] = lower
        res[i, 2] = upper
        i += 1
    plt.plot(ls, res[:, 1], color=color, label=label)
    plt.plot(ls, res[:, 2], color=color)
    plt.fill_between(ls, res[:, 2], res[:, 1], color=color, alpha=0.5)
    plt.plot(ls, res[:, 0], color=mlcolor, label=label)

def plot_errbars_from_sigma_sample_marginals(sigmas,lmax=100, spec=[0,0], sample_fraction=0.68, label=None, color=None, shift=0):
    ls = np.arange(2, lmax + 1)
    res = np.zeros((len(ls), 3))
    res = np.zeros((len(ls), 3))
    i = 0
    for l in ls:
        samps = sigmas[:, l, spec[0], spec[1]] * l * (l+1) / (2 * np.pi)
        x = np.linspace(min(samps), max(samps), 100)
        ml, upper, lower = calculate_ml_and_asymm_errorbars_from_samples(samps, x, sample_fraction, smooth=False)
        res[i, 0] = ml
        res[i, 1] = ml - lower
        res[i, 2] = upper - ml
        i += 1

    ls = ls + shift
    plt.scatter(ls, res[:, 0], color=color, label=label)
    plt.errorbar(ls, res[:, 0], res[:, 1:].T, ecolor=color, fmt=None)

def plot_sigmas_minus_sigmas_by_variance(sigmas, signal, l, spec=[0,0], sample_fraction=0.68, label=None, color=None, nbins=30):
#    ls = np.arange(2, lmax + 1)
    res = np.zeros(3)
    i = 0
#    for l in ls:
    samps = sigmas[:, l, spec[0], spec[1]] * l * (l+1) / (2 * np.pi)
    x = np.linspace(min(samps), max(samps), 100)
    ml, upper, lower = calculate_ml_and_asymm_errorbars_from_samples(samps, x, sample_fraction, smooth=False)
    res[0] = ml
    res[1] = ml - lower
    res[2] = upper - ml
    plt.hist((samps - signal[l, spec[0], spec[1]]*l*(l+1)/2 / np.pi) / (upper-lower), nbins, color=color)
#    ls = ls + shift
#    plt.scatter(ls, res[:, 0], color=color, label=label)
#    plt.errorbar(ls, res[:, 0], res[:, 1:].T, ecolor=color, fmt=None)
