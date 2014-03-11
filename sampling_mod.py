from __future__ import division

import numpy as np
import utils

#Remember that everything is in non-l(l+1) normalization

def sample_constrained_realization(currcls, params):
    constrained_realization = np.zeros((utils.get_num_modes(2, params.lmax+1), np.shape(params.data)[1]))
    for l in range(2, params.lmax+1):
        covmat = np.linalg.inv(np.linalg.inv(currcls[l, :, :]) + params.invnoise[l, :, :])
        mean = params.data[utils.l2mrange(l), :].copy()
        mean = np.dot(np.dot(covmat, params.invnoise), mean[:, :, np.newaxis])[:, :, 0].T
        constrained_realization[utils.l2mrange(l)] = np.random.multivariate_normal(np.zeros(mean.shape[1]), covmat, size=mean.shape[0])
        constrained_realization[utils.l2mrange(l)] += mean
    return constrained_realization

def sample_cls(currsignal, params, prior=None):
    #This depends on the prior. None corresponds to 'flat' prior (though that is not well-defined)
    cls = np.zeros(params.lmax+1)

    for l in range(2, params.lmax+1):
        if prior is None and params.pol:
            #THIS ONLY HOLDS IF dim(cls) = 2 - i.e. no B modes. Otherwise, df = 2l - 3
            df = 2 * l - 2
            cls[l, :, :] = sample_inv_wishart(currsignal[utils.l2mrange(l), :], df, l)
        if prior is None and not params.pol:
            alpha = (2 * l - 1) / 2
            cls[l, :, :] = sample_inv_gamma(currsignal[utils.l2mrange(l), :], alpha)
        #Add other priors here
    return cls

#def sample_inv_wishart(alms, df, l):
def sample_inv_wishart(alms, df):
    #These are really sigma*2l+1, but in the inverse wishart and inverse gamma
    #sampling formalisms, sigma*2l+1 is the 'scaling matrix'
    sigma = np.sum(alms**2, 0)
    invsigma = np.linalg.inv(sigma)

    mean = np.zeros(2)

    samps = np.random.multivariate_normal(mean, invsigma, nu)
    res = np.zeros((sigma.shape))
    for i in range(df):
        res += np.outer(samps[i,:], samps[i, :])
    return np.linalg.inv(res)

#Corresponds to the wikipedia notation
def sample_inv_gamma(alms, alpha):
    beta = np.sum(alms**2, 0)
    return beta / sample_gamma(alpha, 1)

#Corresponds to the wikipedia notation
def sample_gamma(alpha, beta):
    return np.random.gamma(alpha, 1/beta)
