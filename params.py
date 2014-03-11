class Params(object):
    """Stores parameters for the Gibbs sampler"""

    def __init__(self, data, invnoise, init_cls, lmax, polarization=False):
        self.init_cls = init_cls
        self.lmax = lmax
        self.pol = polarization
        self.data = data
        self.invnoise = invnoise
