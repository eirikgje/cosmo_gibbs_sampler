import numpy as np
import utils
import sys

class BlackwellRaoEvaluator(object):
    def __init__(self, method, samples, lmax, init_cls, lmin=4, prior=None):
        self.method = method
        self.samples = np.squeeze(samples.transpose((1, 0, 2, 3)))
        self.lmax = lmax
        self.lmin = lmin
        self.ls = np.arange(lmin, lmax+1)
        if method is inverse_wishart:
            self.halflogsigma2lm1 = 0.5 * (2 * self.ls[:, np.newaxis] - 2) * np.log(owndet(self.samples[self.ls]))
        elif method is inverse_gamma:
            self.halflogsigma2lm1 = 0.5*(2*self.ls[:, np.newaxis] - 1) * np.log(self.samples[self.ls, :])
        elif method is inverse_gamma_marg:
            self.halflogsigma2lm1 = 0.5*(2*self.ls[:, np.newaxis] - 3) * np.log(self.samples[self.ls, :])
        else:
            raise ValueError("Unknown method")
        self.prior = prior
        if self.prior is None:
            self.prior = uniform
#        self.offset = self.get_offset(utils.convert_normalization(init_cls, self.lmax, 'unnormalize'))

    def __call__(self, incls):
        return self.eval_logposterior(incls)

    def eval_logposterior(self, incls):
        cls = np.squeeze(utils.convert_normalization(incls[:self.lmax+1], self.lmax, 'unnormalize'))
        return np.sum(np.log(np.sum(np.exp(self.method(cls[self.ls][:, np.newaxis], self.samples[self.ls, :], self.ls[:, np.newaxis], self.halflogsigma2lm1)), 1)))

    def get_offset(self, cls):
        return 0

class AnalyticEvaluator(object):
    def __init__(self, data, noisespec, lmax, lmin=4, prior=None, pol=False):
        self.data = np.squeeze(utils.convert_normalization(data, lmax, 'unnormalize'))
        self.lmax = lmax
        self.lmin = lmin
        self.prior = prior
        if self.prior is None:
            self.prior = uniform

        self.noisespec = np.squeeze(utils.convert_normalization(noisespec, lmax, 'unnormalize'))
        self.ls = np.arange(lmin, lmax+1)
        self.pol = pol

    def __call__(self, incls):
        if self.pol:
            return self.eval_pol_logposterior(incls)
        else:
            return self.eval_logposterior(incls)

    def eval_logposterior(self, incls):
        cls = np.squeeze(utils.convert_normalization(incls[:self.lmax+1], self.lmax, 'unnormalize'))
        return np.sum(-(2 * self.ls + 1) * 0.5 * (self.data[self.lmin:self.lmax+1] / (cls[self.lmin:self.lmax+1] + self.noisespec[self.lmin:self.lmax+1]) + np.log(cls[self.lmin:self.lmax+1] + self.noisespec[self.lmin:self.lmax+1])))

    def eval_pol_logposterior(self, incls):
        cls = utils.convert_normalization(incls[:self.lmax+1], self.lmax, 'unnormalize')
#        print self.data[self.lmin:self.lmax+1].shape
#        print owninv2(cls[self.lmin:self.lmax+1]+self.noisespec[self.lmin:self.lmax+1]).shape
#        print owndot2(self.data[self.lmin:self.lmax+1], owninv2(cls[self.lmin:self.lmax+1] + self.noisespec[self.lmin:self.lmax+1])).shape
#        sys.exit()
        return np.sum(-(2*self.ls +1) * 0.5 * (np.trace(owndot2(self.data[self.lmin:self.lmax+1], owninv2(cls[self.lmin:self.lmax+1] + self.noisespec[self.lmin:self.lmax+1])), axis1=1, axis2=2) + np.log(owndet2(cls[self.lmin:self.lmax+1] + self.noisespec[self.lmin:self.lmax+1]))))

def inverse_wishart(cls, sigma, l, sigmaterm):
    return -0.5*(2 * l +1) * np.log(owndet(cls)) - 0.5 * (2*l+1) * np.trace(owndot(sigma, owninv(cls)), axis1=2, axis2=3) + sigmaterm

def inverse_gamma(cls, sigma, l, halflogsigma2lm1):
    return -0.5*(2*l+1) * (np.log(cls) +sigma/cls) + halflogsigma2lm1

def inverse_gamma_marg(cls, sigma, l, halflogsigma2lm1):
    return -0.5*(2*l+1) * sigma/cls - 0.5*(2*l-1) * np.log(cls) + halflogsigma2lm1

def uniform(cls, l):
    return 0.0

def jeffreys(cls, l):
    return -np.log(cls)

def owndet(mats):
    return mats[:, :, 0, 0] * mats[:, :, 1, 1] - mats[:, :, 0, 1] * mats[:, :, 1, 0]

def owndet2(mats):
    return mats[:, 0, 0] * mats[:, 1, 1] - mats[:, 0, 1] * mats[:, 1, 0]

def owninv(mats):
    res = np.zeros(np.shape(mats))
    res[:,:,  0, 0] = mats[:,:,  1, 1]
    res[:,:,  1, 1] = mats[:,:,  0, 0]
    res[:,:,  0, 1] = -mats[:,:,  0, 1]
    res[:,:,  1, 0] = -mats[:,:,  1, 0]
    return res / owndet(mats)[:, :, np.newaxis, np.newaxis]

def owninv2(mats):
    res = np.zeros(np.shape(mats))
    res[:,  0, 0] = mats[:,  1, 1]
    res[:,  1, 1] = mats[:,  0, 0]
    res[:,  0, 1] = -mats[:,  0, 1]
    res[:,  1, 0] = -mats[:,  1, 0]
    return res / owndet2(mats)[:, np.newaxis, np.newaxis]

def owndot(mat1, mat2):
    return np.sum(np.swapaxes(mat1, 2, 3)[:, :, np.newaxis, :, :] * mat2[:, :, :, :, np.newaxis], 3)

def owndot2(mat1, mat2):
    return np.sum(np.swapaxes(mat1, 1, 2)[:, np.newaxis, :, :] * mat2[:, :, :, np.newaxis], 2)
