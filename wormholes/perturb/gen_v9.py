"""
    Like V3 (demo V1) but with the same image-target pairs across model and budgets, a narrower budget regime, and start images of type ANI2
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v6 import GenV6


class GenV9(GenV6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Drop robust model 1.0 & vanilla
        self.model_kwargs_dict = {k: v for k, v in self.model_kwargs_dict.items() if k not in ['resnet50_robust_mapped_RIN_l2_1_0',
                                                                                               'resnet50_vanilla_mapped_RIN']}
        self.model_subjects_names = np.unique(chained(list(x['model_subjects'].keys()) for x in self.model_kwargs_dict.values()))
        self.data_ANI = glob.glob(f"{self.data_root}/ANI2/*")
        self.attack_hparams_tup_list = [namedtuple('attack_hparams', ['eps', 'step_size', 'n_iter'])(*x) 
                                        for x in [(300, 4, 10_000), (150, 2, 10_000), (100, 2, 5000), (50, 2, 2000), (30, 2, 2000), (20, .5, 1000), 
                                                  (0., 0., 0)
                                                  ]]
        
    def get_data(self):
        # Arbitrary Natural Images
        triplet_paths_list = [[(img_path[len(f"{self.data_root}/"):], 'ANI'), target_class_name] 
                              for img_path in np.random.choice(self.data_ANI, size=25, replace=False) 
                              for target_class_name in self.data_dict]
        return triplet_paths_list
