from __future__ import division

import numpy as np
import slicer
import likelihood_mod as lhm
import utils
reload(slicer)
reload(lhm)

dims = 1
lmax= 100
lmin = 4
nbins = 20
ls = np.arange(0, lmax+1)
lpivot = 50

data = utils.convert_normalization(utils.alms2sigma(np.load('data_70t_20p_60tp.npy')[:utils.get_num_modes(0, lmax), :dims]), lmax)
noise = utils.convert_normalization(np.array([np.load('noise_70t_20p_60tp.npy')[:dims, :dims]]*(lmax+1)), lmax)

samps = np.load('tsamps.npy')[:, :lmax+1]
ordering = {'TT':1, 'EE':2, 'TE':4}
cls = utils.cls2matrix_all(np.loadtxt('camb_91836017_scalcls_with_tau0.09_r0.1.dat'), ordering, dims)
cls = np.append(np.zeros((2, dims, dims)), cls, 0)[:lmax+1, :, :]

an_like = lhm.AnalyticEvaluator(data, noise, lmax, lmin=lmin, pol=False)
br_like = lhm.BlackwellRaoEvaluator(lhm.inverse_gamma, samps, lmax, cls, lmin=lmin)

anslicer = slicer.Slicer_2d(an_like, numbin_twopar=nbins, lpivot=lpivot)
brslicer = slicer.Slicer_2d(br_like, numbin_twopar=nbins, lpivot=lpivot)
Qbounds, nbounds, probs = anslicer.run_slicer()
Qbounds_br, nbounds_br, probs_br = brslicer.run_slicer()
dQ = (Qbounds[1] - Qbounds[0]) / (nbins - 1)
dn = (nbounds[1] - nbounds[0]) / (nbins - 1)

#brslicer = slicer.Slicer_2d(

with file('probs_test1_an.dat', 'w') as outfile:
    np.savetxt(outfile, np.array([nbins, nbins]).reshape(1, 2), fmt='%2d')
    for i in range(nbins):
        Q = Qbounds[0] + dQ * i
        for j in range(nbins):
            n = nbounds[0] + dn * j
            np.savetxt(outfile, np.array([Q, n, probs[i,j]]).reshape(1, 3))

dQ = (Qbounds_br[1] - Qbounds_br[0]) / (nbins - 1)
dn = (nbounds_br[1] - nbounds_br[0]) / (nbins - 1)

with file('probs_test1_br.dat', 'w') as outfile:
    np.savetxt(outfile, np.array([nbins, nbins]).reshape(1, 2), fmt='%2d')
    for i in range(nbins):
        Q = Qbounds_br[0] + dQ * i
        for j in range(nbins):
            n = nbounds_br[0] + dn * j
            np.savetxt(outfile, np.array([Q, n, probs_br[i,j]]).reshape(1, 3))
