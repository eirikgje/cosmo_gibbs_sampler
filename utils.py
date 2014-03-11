from __future__ import division

import numpy as np

def get_noise_powerspec(rms, lmax=47, nside=16, beam_fwhm=None, pol=False):
    "FWHM in arcmin"
    l = np.arange(0, lmax + 1)
    npix = 12 * nside ** 2
    spec = 4 * np.pi / npix * rms ** 2 * l * (l + 1) / (2 * np.pi)
    if beam_fwhm is not None:
        sigma = beam_fwhm/(60.0 * 180.0) * np.pi / np.sqrt(8.0*np.log(2.0))
        sig2 = sigma ** 2
        g = np.exp(-0.5*l*(l+1.0)*sig2)
        if pol:
            factor_pol = np.exp([0.0, 2.0*sig2, 2.0*sig2])
            gout = g * np.exp(2.0 *sig2)
        else:
            gout = g
        spec = spec / g**2
    return spec

def l2mrange(l):
    start = l**2 - (l-1)**2 - 3
    end = start + (l+1)**2 - l ** 2
    return slice(start, end)

def get_num_modes(lmin, lmax):
    #Find number of modes between lmin and lmax, inclusive
    return (lmax+1) ** 2 - lmin**2
