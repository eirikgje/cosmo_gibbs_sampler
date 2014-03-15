from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

#def get_noise_powerspec(rms, lmax=47, nside=16, beam_fwhm=None, pol=False):
#    "FWHM in arcmin"
#    l = np.arange(0, lmax + 1)
#    npix = 12 * nside ** 2
#    spec = 4 * np.pi / npix * rms ** 2 * l * (l + 1) / (2 * np.pi)
#    spec = 4 * np.pi / npix * rms ** 2
#    if beam_fwhm is not None:
#        sigma = beam_fwhm/(60.0 * 180.0) * np.pi / np.sqrt(8.0*np.log(2.0))
#        sig2 = sigma ** 2
#        g = np.exp(-0.5*l*(l+1.0)*sig2)
#        if pol:
#            factor_pol = np.exp([0.0, 2.0*sig2, 2.0*sig2])
#            gout = g * np.exp(2.0 *sig2)
#        else:
#            gout = g
#        spec = spec / g**2
#    return spec
def get_noise_powerspec(lmax, value, dim):
    noise = np.zeros((lmax+1, dim, dim))
    noise[:, :, :] = value[np.newaxis, :, :]
    return noise

def l2mrange(l):
    start = 0
    start = get_num_modes(0, l-1)
    end = start + get_num_modes(l, l)
#    for currl in xrange(l):
#        start += get_num_modes(currl, currl)
#    end = start + get_num_modes(l, l)
    return slice(start, end)

def get_num_modes(lmin, lmax):
    #Find number of modes between lmin and lmax, inclusive
    return (lmax+1) ** 2 - lmin**2

def cls2matrix(cl, ordering, dim):
    s = np.zeros((dim, dim))
    s[0, 0] = cl[ordering['TT']]
    if dim == 1:
        return s
    s[1, 0] = cl[ordering['TE']]
    s[0, 1] = s[1, 0]
    s[1, 1] = cl[ordering['EE']]
    if dim == 2:
        return s
    s[2, 0] = cl[ordering['TB']]
    s[0, 2] = s[2, 0]
    s[2, 1] = cl[ordering['EB']]
    s[1, 2] = s[2, 1]
    s[2, 2] = cl[ordering['BB']]
    return s

def cls2matrix_all(cl, ordering, dim):
    s = np.zeros((cl.shape[0], dim, dim))
    for l in xrange(cl.shape[0]):
        s[l, :, :] = cls2matrix(cl[l, :], ordering, dim)
    return s

def alms2sigma(alms):
    lmax = int(np.sqrt(alms.shape[0]) - 1)
    sigma = np.zeros((lmax+1, alms.shape[1], alms.shape[1]))
    for l in range(lmax+1):
        for i in range(alms.shape[1]):
            for j in range(alms.shape[1]):
                sigma[l, i, j] = np.sum(alms[l2mrange(l), i]*alms[l2mrange(l), j]) / (2 * l + 1)
    return sigma

def convert_normalization(cls, lmax, mode='normalize'):
    ls = np.arange(lmax+1)
    #not used anyway
    ls[0] = 1.0
    if mode == 'normalize':
        factor = ls * (ls+1) / (2 * np.pi)
    elif mode == 'unnormalize':
        factor = 2 * np.pi / (ls * (ls + 1))

    return cls*factor[:, np.newaxis, np.newaxis]

def get_inverse_wishart_1d(newnu, scale, x):
    return scale**(newnu / 2) / (2 **(newnu / 2)) / gamma(newnu/2) / x ** ((newnu + 2) / 2) * np.exp(-0.5 * scale / x)
