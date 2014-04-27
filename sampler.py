class GibbsSampler(object):
    """The object doing the actual sampling."""

    def __init__(self, params, signal_sampler, cl_sampler, init_cls, prior=None, output_routine=None):
        self.currcls = params.init_cls
        self.signal_sampler = signal_sampler
        self.cl_sampler = cl_sampler
        self.init_cls = init_cls
        self.curralms = None
        self.prior = prior
        self.params = params
        self.output_routine=output_routine

    def step_gibbs(self):
        self.curralms = self.signal_sampler(self.currcls, self.params)
        self.currcls = self.cl_sampler(self.curralms, self.params)
        return self.curralms, self.currcls

    def output_sample(self):
        self.output_routine(self.currcls, self.curralms)
