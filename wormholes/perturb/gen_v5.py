"""
    Like V3 (V1 demo mode) but with denser sampling and 12x more images
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v3 import GenV3


class GenV5(GenV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_hparams_tup_list = [namedtuple('attack_hparams', ['eps', 'step_size', 'n_iter'])(*x) 
                                        for x in [(300, 4, 10_000), (200, 3, 10_000), (150, 2, 10_000), 
                                                  (100, 2, 5000), (50, 2, 2000), (40, 2, 2000), (30, 2, 2000), (25, .5, 1000), (20, .5, 1000), 
                                                  (15, .5, 500), (10, .5, 500), (7.5, .5, 500), (5, .5, 500), 
                                                  (3., .3, 200), (2., .3, 200), (1., .3, 200), (.5, .1, 200), (.1, .02, 200),
                                                  (0., 0., 0)
                                                  ]]
        
        self.interp_hparams_tup_list = [namedtuple('interp_hparams', ['eps', 'alpha_interp'])(*x) 
                               for x in itertools.product([300, 200, 150, 100, 50, 40, 30, 25, 20, 15, 10, 7.5, 5., 3., 2., 1., .5, .1], 
                                                          [0.])]

    def get_data(self, n_sample_class=12):
        return super().get_data(n_sample_class=n_sample_class)

    