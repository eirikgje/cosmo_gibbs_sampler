import params
import numpy as np
from sampler import GibbsSampler
import utils
import sampling_mod
import matplotlib.pyplot as plt

def plot_sample(cls, alms):
    sigma = utils.alms2sigma(alms)
    ncls = utils.convert_normalization(cls, lmax, 'normalize')
    sigma = utils.convert_normalization(sigma, lmax, 'normalize')
    plt.figure(1)
    plt.plot(ncls[:, 0, 0], color='blue')
    plt.plot(sigma[:, 0, 0], color='green')
    plt.figure(2)
    plt.plot(ncls[:, 1, 0], color='blue')
    plt.plot(sigma[:, 1, 0], color='green')
    plt.figure(3)
    plt.plot(ncls[:, 1, 1], color='blue')
    plt.plot(sigma[:, 1, 1], color='green')

l_noise_t = 70
l_noise_p = 20
l_noise_tp = 60
lmax = 100
dims = 1
load_dat = True
pol = False

cls = np.loadtxt('camb_91836017_scalcls_with_tau0.09_r0.1.dat')
#cls = np.zeros(precls.shape[0]+2, 2, 2)
ordering = {'TT':1, 'EE':2, 'TE':4}
cls = utils.cls2matrix_all(cls, ordering, dims)
cls = np.append(np.zeros((2, dims, dims)), cls, 0)[:lmax+1, :, :]
#for l in range(2, precls.shape[0]+2):
#    cls[l, :] = utils.cls2matrix(precls[l-2, :], ordering, 2)
cls = utils.convert_normalization(cls, lmax, 'unnormalize')
if load_dat:
#    noise = np.load('noise_4times.npy')[:dims, :dims]
#    data = np.load('data_4times.npy')[:, :dims]
    noise = np.load('noise_70t_20p_60tp.npy')[:dims, :dims]
    data = np.load('data_70t_20p_60tp.npy')[:, :dims]
#    noise = np.load('noise.npy')[:dims, :dims]
#    data = np.load('data.npy')[:, :dims]
else:
#    alms = sampling_mod.sample_alms(cls)
    alms = np.load('signal.npy')
#    np.save('signal.npy', alms)
    sigma = utils.alms2sigma(alms)
#    noise = sigma[l_noise]
    noise = sigma[l_noise_tp].copy()
    noise[0, 0] = sigma[l_noise_t, 0, 0].copy()
    noise[1, 1] = sigma[l_noise_p, 1, 1].copy()
#    noise *= 4
#    np.save('noise_4times.npy', noise)
    np.save('noise_70t_20p_60tp.npy', noise)
    data = alms + sampling_mod.sample_white_noise(noise, lmax, dims)
    np.save('data_70t_20p_60tp.npy', data)

invn = np.linalg.inv(noise)
invn = np.array([invn]*(lmax+1))

pars = params.Params(data, invn, cls, lmax, pol)

sampler = GibbsSampler(pars, sampling_mod.sample_constrained_realization, sampling_mod.sample_cls, cls, output_routine=plot_sample)

cls = [cls]
sigmas = []

#with file('sigma_pol_wishart_70t_20p_60tp.dat', 'w') as sigmaout:
#    with file('cls_pol_wishart_70t_20p_60tp.dat', 'w') as clout:
with file('sigma_temp_gamma_70t.dat', 'w') as sigmaout:
    with file('cls_temp_gamma_70t.dat', 'w') as clout:
        for i in xrange(100000):
            if i % 100 == 0:
                print i
            a, c = sampler.step_gibbs()
            cls.append(c)
            sigmas.append(utils.alms2sigma(a))
#            sampler.output_sample()
            np.savetxt(clout, sampler.currcls.flatten())
            np.savetxt(sigmaout, sigmas[-1].flatten())
#            plt.show()
