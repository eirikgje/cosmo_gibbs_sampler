import numpy as np

class BlackwellRaoEvaluator(object):
    def __init__(self, method, samples, lmax, init_cls, lmin=4, prior=None):
        self.method = method
        self.samples = samples.transpose((1, 0, 2, 3))
        self.lmax = lmax
        self.lmin = lmin
        self.ls = np.arange(lmin, lmax+1)
        self.prior = prior
        if self.prior is None:
            self.prior = uniform
        self.offset = self.get_offset(cls)

    def eval_logposterior(self, cls):
        return np.sum(np.exp(np.sum([self.method(cl, self.samples[l], l) for cl, l in zip(cls, self.ls)], 0) - self.offset))*self.prior(cl, l)

    def get_offset(self, cls):
        return np.max(np.sum([self.method(cls, self.samples, l) for cl, l in zip(cls, self.ls)], 0))

def inverse_wishart(cls, sigma, l):
    return -0.5*(2 * l +1) * np.log(np.linalg.det(cls)) - 0.5 * (2*l+1) * np.trace(np.dot(sigma, np.linalg.inv(cls)), 1, 2)

def inverse_gamma(cls, sigma, l):
    return (-0.5*(2*l+1) * np.log(cls) - 0.5*(2*l+1) * sigma / cls)[:, 0, 0]

def uniform(cls, l):
    return 1.0
