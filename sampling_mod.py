from __future__ import division

import numpy as np
import utils

#Remember that everything is in non-l(l+1) normalization

def sample_constrained_realization(currcls, params):
    constrained_realization = np.zeros((utils.get_num_modes(0, params.lmax), np.shape(params.data)[1]))
    for l in range(3, params.lmax+1):
        covmat = np.linalg.inv(np.linalg.inv(currcls[l, :, :]) + params.invnoise[l, :, :])
        mean = params.data[utils.l2mrange(l), :].copy()
        mean = np.dot(np.dot(covmat, params.invnoise[l, :, :]), mean[:, :, np.newaxis])[:, :, 0].T
        constrained_realization[utils.l2mrange(l), :] = np.random.multivariate_normal(np.zeros(mean.shape[1]), covmat, size=mean.shape[0])
        constrained_realization[utils.l2mrange(l), :] += mean
    return constrained_realization

def sample_cls(currsignal, params, prior=None):
    #This depends on the prior. None corresponds to 'flat' prior (though that is not well-defined)
    cls = np.zeros((params.lmax+1, currsignal.shape[1], currsignal.shape[1]))

    for l in range(3, params.lmax+1):
        if prior is None and params.pol:
            #THIS ONLY HOLDS IF dim(cls) = 2 - i.e. no B modes. Otherwise, df = 2l - 3
            df = 2 * l - 2
            cls[l, :, :] = sample_inv_wishart(currsignal[utils.l2mrange(l), :], df)
        if prior is None and not params.pol:
            alpha = (2 * l - 1) / 2
            cls[l, :, :] = sample_inv_gamma(currsignal[utils.l2mrange(l), :], alpha)
        #Add other priors here
    return cls

#def sample_inv_wishart(alms, df, l):
def sample_inv_wishart(alms, df):
    #These are really sigma*2l+1, but in the inverse wishart and inverse gamma
    #sampling formalisms, sigma*2l+1 is the 'scaling matrix'
#    sigma = np.sum(alms**2, 0)
    sigma = np.zeros((alms.shape[1], alms.shape[1]))
    for i in xrange(alms.shape[1]):
        for j in xrange(alms.shape[1]):
            sigma[i, j] = np.sum(alms[:, i]*alms[:, j])
    invsigma = np.linalg.inv(sigma)

    mean = np.zeros(2)

    samps = np.random.multivariate_normal(mean, invsigma, df)
    res = np.zeros((sigma.shape))
    for i in range(df):
        res += np.outer(samps[i,:], samps[i, :])
    return np.linalg.inv(res)

#Corresponds to the wikipedia notation
def sample_inv_gamma(alms, alpha):
    beta = 0.5 * np.sum(alms**2, 0)
    return beta / sample_gamma(alpha, 1)
#    return 1 / sample_gamma(alpha, 1/beta)
#    return 1.0 / sample_gamma(alpha, beta)

#Corresponds to the wikipedia notation
def sample_gamma(alpha, beta):
    return np.random.gamma(alpha, 1.0 / beta)

def sample_alms(cls):
    alms = np.zeros((utils.get_num_modes(0, cls.shape[0]-1), cls.shape[1]))
    mean = np.zeros(cls.shape[1])
    for l in range(3, cls.shape[0]):
        alms[utils.l2mrange(l), :] = np.random.multivariate_normal(mean, cls[l], size=2*l+1)
    return alms

def sample_white_noise(val, lmax, dim):
    mean = np.zeros(dim)
    return np.random.multivariate_normal(mean, val, size=utils.get_num_modes(0, lmax))

