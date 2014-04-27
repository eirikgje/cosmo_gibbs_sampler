from __future__ import division

import likelihood_mod as lhm
import numpy as np
import utils

dims = 1
lmax= 70
lmin = 70
nbins = 1000
ls = np.arange(0, lmax+1)
reload(lhm)

#data = utils.convert_normalization(utils.alms2sigma(np.load('data.npy')[:utils.get_num_modes(0, lmax), :dims]), lmax)
#noise = utils.convert_normalization(np.array([np.load('noise.npy')[:dims, :dims]]*(lmax+1)), lmax)
data = utils.convert_normalization(utils.alms2sigma(np.load('data_4times.npy')[:utils.get_num_modes(0, lmax), :dims]), lmax)
noise = utils.convert_normalization(np.array([np.load('noise_4times.npy')[:dims, :dims]]*(lmax+1)), lmax)
#samps = utils.sampfile_to_samples('sigma_temp_gamma_100k_4noise.dat', 100, dims)[:lmax+1]
#samps = utils.sampfile_to_samples('sigma_pol_wishart_100k_tnoise.dat', 100, dims)[:lmax+1]
#samps = np.load('psamps.npy')[:, :lmax+1, :dims, :dims]
samps = np.load('tsamps.npy')[:, :lmax+1]

ordering = {'TT':1, 'EE':2, 'TE':4}
cls = utils.cls2matrix_all(np.loadtxt('camb_91836017_scalcls_with_tau0.09_r0.1.dat'), ordering, dims)
cls = np.append(np.zeros((2, dims, dims)), cls, 0)[:lmax+1, :, :]

an_like = lhm.AnalyticEvaluator(data, noise, lmax, lmin=lmin)
br_like = lhm.BlackwellRaoEvaluator(lhm.inverse_gamma, samps, lmax, cls, lmin=lmin)
#br_like = lhm.BlackwellRaoEvaluator(lhm.inverse_gamma_marg, samps, lmax, cls, lmin=lmin)
#br_like = lhm.BlackwellRaoEvaluator(lhm.inverse_wishart, samps, lmax, cls, lmin=lmin)

loglikes = np.zeros(nbins)
loglikes_br = np.zeros(nbins)
#likes_br = np.zeros(nbins)
As = np.zeros(nbins)
Amin = 0.1
Amax = 8.0
dA = (Amax - Amin) / (nbins - 1)
for i in xrange(nbins):
    if i % 10 == 0: print i
    currcls = (Amin + dA * i) * cls
#    As[i] = currcls[lmin]
    As[i] = Amin + dA*i
    loglikes_br[i] = br_like.eval_logposterior(currcls)
#    likes_br[i] = br_like.eval_logposterior(currcls)
    loglikes[i] = an_like.eval_logposterior(currcls)
#    loglikes[i] = an_like.eval_pol_logposterior(currcls)
    print loglikes_br[i]

likes = np.exp(loglikes - np.max(loglikes))
likes = likes / (np.sum(likes) * (As[1] - As[0]))
likes_br = np.exp(loglikes_br - np.max(loglikes_br))
likes_br = likes_br / (np.sum(likes_br) * (As[1]- As[0]))
