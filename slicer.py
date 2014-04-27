from __future__ import division

import numpy as np
from scipy.optimize import fmin_powell
import sys

#Module to calculate 2d slices
#ref_cl_name = '/mn/stornext/u2/eirikgje/data/vault/general/bestfit_lensedCls.dat'
ref_cl_name = 'camb_91836017_scalcls_with_tau0.09_r0.1.dat'
#ref_cl_name = '/mn/stornext/u2/eirikgje/src/wmap_likelihood_v5/data/test_cls_v5.dat'

class Slicer_2d(object):
    def __init__(self, evaluator, Q_init=1.0, n_init=0.0, ref_cls=None, numbin_twopar=20, lpivot=40, dims=1):
        self.Q = Q_init
        self.n = n_init
        self.numbin_twopar=numbin_twopar
        self.ref_cls=ref_cls
        self.lpivot = lpivot
        self.evaluator = evaluator
        self.dims = dims

    def minlogl(self, Qn):
        cls = self.get_parametrized_cls(Qn)
        return -self.evaluator(cls)

    def find_maxpoint(self, initial_guess):
        vals = np.zeros(2)
        vals[0], vals[1] = fmin_powell(self.minlogl, initial_guess)
        minloglike = self.minlogl(vals)
        print 'maxpoint', vals
        return vals, minloglike

    def get_bounds(self, Q_in, n_in, maxlogl):
        dlnL = 20.0
        start_point = np.array([Q_in, n_in])
        delta = np.zeros(2)
        lnL0 = maxlogl
        bounds = {'Q':[0.0, 0.0], 'n':[0.0, 0.0]}
    
        #Find parameter bounds
        for i, (par, parname) in enumerate(zip(start_point, ('Q', 'n'))):
            delta[:] = 0
            delta[i] = np.max(0.001, 0.01*np.abs(par))
            lnL = -self.minlogl(start_point - delta)
            while lnL0 - lnL < dlnL:
                delta *= 2
                lnL = -self.minlogl(start_point - delta)
            bounds[parname][0] = par - delta[i]

            delta[:] = 0
            delta[i] = np.max(0.001, 0.01*np.abs(par))
            lnL = -self.minlogl(start_point + delta)
            while lnL0 - lnL < dlnL:
                delta *= 2
                lnL = -self.minlogl(start_point + delta)
            bounds[parname][1] = par + delta[i]

        return bounds

    def get_parametrized_cls(self, Qn):
        ls = np.arange(self.ref_cls.shape[0])
        #Doesn't matter - the l=0 isn't used anyway.
        ls[0] = 1.0
        return Qn[0] * self.ref_cls * (ls[:, np.newaxis, np.newaxis] / float(self.lpivot)) ** Qn[1]

    def run_slicer(self):
        if self.ref_cls is None:
            self.ref_cls = load_default_refcls(self.dims)

        initial_guess = [self.Q, self.n]
    
        print 'Finding maxpoint'
        vals, minloglike = self.find_maxpoint(initial_guess)
        self.Q = vals[0]
        self.n = vals[1]
        self.maxlogl = -minloglike
    
        print 'Getting bounds'
        bounds = self.get_bounds(self.Q, self.n, self.maxlogl)

        dQ = (bounds['Q'][1] - bounds['Q'][0]) / (self.numbin_twopar - 1)
        dn = (bounds['n'][1] - bounds['n'][0]) / (self.numbin_twopar - 1)

        print 'Mapping out contours'
        prob = np.zeros((self.numbin_twopar, self.numbin_twopar))
        for i in range(self.numbin_twopar):
            Q = bounds['Q'][0] + i * dQ
            for j in range(self.numbin_twopar):
                n = bounds['n'][0] + j * dn
                prob[i, j] = -self.minlogl([Q, n])

        self.prob = prob
        return bounds['Q'], bounds['n'], np.exp(prob -np.max(prob))

def load_default_refcls(dims):
    temp = np.loadtxt(ref_cl_name).T
    ref_cls = np.zeros((2+temp.shape[1], dims, dims))
    ref_cls[2:, 0, 0] = temp[1, :]
    if dims == 2:
        ref_cls[2:, 1, 1] = temp[2, :]
        ref_cls[2:, 1, 0] = temp[4, :]
        ref_cls[2:, 0, 1] = temp[4, :]

    return ref_cls
