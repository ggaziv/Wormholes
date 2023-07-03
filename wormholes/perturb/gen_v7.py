"""
    Implementation of RestrictedImageNet-based category modulation by attacking model logits
    With linf attacks and linf-trained models
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v6 import GenV6


class GenV7(GenV6):
    attack_hparams_tup_list_l2 = [namedtuple('attack_hparams', ['eps', 'step_size', 'n_iter', 'constraint'])(*x, '2') 
                                  for x in [(300, 4, 10_000), (200, 3, 10_000), (150, 2, 10_000), 
                                            (100, 2, 5000), (50, 2, 2000), (30, .5, 1000), 
                                            (10, .5, 500), (5, .5, 500), 
                                            (2.5, .3, 200), (1., .3, 200), (.5, .1, 200), (.1, .02, 200),
                                            (0., 0., 0)]]
    attack_hparams_tup_list_linf255 = [namedtuple('attack_hparams', ['eps', 'step_size', 'n_iter', 'constraint'])(*x, 'inf') 
                                    for x in [(250, 4, 10_000), (150, 4, 10_000), (128, 4, 10_000), 
                                              (64, 3, 1000), 
                                              (32, 3, 500), 
                                              (16, 2, 200), (8, 1.2, 200), (4, .2 ,200), 
                                              (0., 0., 0)]] 
    
    attack_hparams_tup_list = attack_hparams_tup_list_l2 + attack_hparams_tup_list_linf255
    
    interp_hparams_tup_list = [namedtuple('interp_hparams', ['eps', 'alpha_interp', 'constraint'])(*x) 
                                        for x in (list(itertools.product([300, 200, 150, 100, 50, 30, 10, 5, 2.5, 1., .5, .1], [0.], ['2'])) + 
                                                  list(itertools.product([250, 150, 128, 64, 32, 16, 8, 4], [0.], ['inf'])))]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_dict = {
            'model_name': str,
            'budget': float, 'n_iter': int, 'step_size': float, 'interp_alpha': float, 'constraint': str,
            'target_class_name': str,
            'orig_class_name': str,
            'orig_name': str,
            'triplet': int,
            'budget_usage': float, 
            'target_image_path': str,
            }
        self.group_columns = ['model_name', 'budget', 'n_iter', 'step_size', 'interp_alpha', 'constraint']
        ds = RestrictedImageNet(f"{self.data_root}/ilsvrc")
        get_mapped_RIN = lambda resume_name: get_restricted_imagenet_mapped_model(arch='resnet50', 
                                                                                  restricted_imagenet_ds=ds, 
                                                                                  pytorch_pretrained=False, 
                                                                                  resume_path=f"{PROJECT_ROOT}/checkpoints/{resume_name}")[0].eval()
        self.model_kwargs_dict = {
            'resnet50_vanilla_mapped_RIN': {
                'attacker':
                    lambda : get_restricted_imagenet_mapped_model(
                        arch='resnet50', 
                        restricted_imagenet_ds=ds, 
                        pytorch_pretrained=True, 
                        resume_path=None)[0],
                    'model_subjects': {
                        'resnet50_vanilla_mapped_RIN_v2': partial(get_mapped_RIN, 'imagenet_vanilla_v2.pt')
                        } | {
                            f'resnet50_robust_mapped_RIN_l2_{eps_eval}_0_v2': partial(get_mapped_RIN, f'imagenet_l2_{eps_eval}_0_v2.pt') 
                            for eps_eval in [1, 3, 10]    
                        } | {
                            f'resnet50_robust_mapped_RIN_linf_{eps_eval}': partial(get_mapped_RIN, f'imagenet_linf_{eps_eval}.pt') 
                            for eps_eval in [4, 8]    
                            }
                    }
        } | {
            f'resnet50_robust_mapped_RIN_l2_{eps}_0': {
                'attacker':
                    partial(get_mapped_RIN, f'imagenet_l2_{eps}_0.pt'),
                    'model_subjects': { 
                        f'resnet50_robust_mapped_RIN_l2_{eps_eval}_0_v2': partial(get_mapped_RIN, f'imagenet_l2_{eps_eval}_0_v2.pt') 
                        for eps_eval in [1, 3, 10]
                        } | {
                            'resnet50_vanilla_mapped_RIN_v2': partial(get_mapped_RIN, 'imagenet_vanilla_v2.pt')
                        } | {
                            f'resnet50_robust_mapped_RIN_linf_{eps_eval}': partial(get_mapped_RIN, f'imagenet_linf_{eps_eval}.pt') 
                            for eps_eval in [4, 8]    
                            }
                    } for eps in [1, 3, 10]
        } | {
            f'resnet50_robust_mapped_RIN_linf_{eps}': {
                'attacker':
                    partial(get_mapped_RIN, f'imagenet_linf_{eps}.pt'),
                    'model_subjects': { 
                        f'resnet50_robust_mapped_RIN_l2_{eps_eval}_0_v2': partial(get_mapped_RIN, f'imagenet_l2_{eps_eval}_0_v2.pt') 
                        for eps_eval in [1, 3, 10]
                    } | {
                        'resnet50_vanilla_mapped_RIN_v2': partial(get_mapped_RIN, 'imagenet_vanilla_v2.pt')
                    } | {
                        f'resnet50_robust_mapped_RIN_linf_{eps_eval}': partial(get_mapped_RIN, f'imagenet_linf_{eps_eval}.pt') 
                        for eps_eval in [4, 8]    
                        }
                    } for eps in [4, 8]
            }
        # Add the self-subject to all attackers
        for model_name, d in self.model_kwargs_dict.items():
            d['model_subjects'][model_name] = d['attacker']
            
        self.contrast_blend_model_subjects = {
            'resnet50_vanilla_mapped_RIN_v2': partial(get_mapped_RIN, 'imagenet_vanilla_v2.pt')
            } | {
                f'resnet50_robust_mapped_RIN_l2_{eps_eval}_0_v2': partial(get_mapped_RIN, f'imagenet_l2_{eps_eval}_0_v2.pt') 
                for eps_eval in [1, 3, 10]    
            } | {
                f'resnet50_robust_mapped_RIN_linf_{eps_eval}': partial(get_mapped_RIN, f'imagenet_linf_{eps_eval}.pt') 
                for eps_eval in [4, 8]    
                }
            

        folder_ds = ImageFolder(root=f"{ds.data_path}/val", label_mapping=ds.label_mapping)
        data_dict = invert_dict(dict(folder_ds.samples))
        k = min([len(v) for v in data_dict.values()])
        self.data_dict = {self.class_dict[int(lbl)]: v[:k] for lbl, v in data_dict.items()}
        self.class_name_to_index = {v: k for k, v in self.class_dict.items()}
        
        self.transform = transforms.Compose([transforms.ToPILImage()] + 
                                            ds.transform_test.transforms[:2] + 
                                            [np.array])
        
        self.model_subjects_names = np.unique(chained(list(x['model_subjects'].keys()) for x in self.model_kwargs_dict.values()))
        self.output_folder = f"{PROJECT_ROOT}/results/cache/gen_v{self.gen_version}"
        
    def run(self):
        args = self.args
          
        self.rng_job = np.random.default_rng(int(args.seed or 0))
              
        configs = list(itertools.product(self.model_kwargs_dict.items(), self.attack_hparams_tup_list)) + self.interp_hparams_tup_list
        cprintm(f"Total number of configs: {len(configs)}")
        
        ds = self.get_ds()
        df = self.__class__.dataset_to_dataframe(ds)
        
        # Screen for current job
        group, df_agg = self.screen_job_work(df)

        self.make_and_save_results(ds, df_agg, group, 
                                   batch_size=args.batch_size)
    
    
    def add_contrast_blend(self, meta_list, gen):
        for hp, triplet_index, target_class_name, orig_class_name in gen(self.interp_hparams_tup_list):
            row = [
                'contrast_blend', 
                hp.eps, 0, np.nan, hp.alpha_interp, hp.constraint,
                target_class_name,
                orig_class_name,
                np.nan,
                triplet_index,
                ]
            meta_list.append(row + [np.nan] * (len(self.field_dict) - len(row)))
                         
    def add_attack(self, meta_list, gen):
        for model_name in self.model_kwargs_dict:
            for hp, triplet_index, target_class_name, orig_class_name in gen(self.attack_hparams_tup_list):
                row = [
                    model_name, 
                    hp.eps, hp.n_iter, hp.step_size, np.nan, hp.constraint,
                    target_class_name,
                    orig_class_name,
                    np.nan,
                    triplet_index,
                    ]
                meta_list.append(row + [np.nan] * (len(self.field_dict) - len(row)))
                    
    def make_and_save_results(self, ds, df_agg, g,  
                              batch_size=50):  
        meta_filename = f"meta_job{self.args.num}.nc"
        df_chunks = list(chunks(df_agg, batch_size))
        if np.isnan(g.interp_alpha):
            attacker_model = self.make_attacker_model(g.model_name)
        budget = g.budget 
        step_size = g.step_size
        if g.constraint == 'inf':
            budget, step_size = [x / 255. for x in [budget, step_size]]
        for cnk_i, cnk in enumerate(df_chunks):
            cprint1(f"chunk [{cnk_i+1}/{len(df_chunks)}]")
            source_images_paths = [self.rng_job.choice(self.data_dict[class_name]) for class_name in cnk.orig_class_name]
            images_source = self.get_images(source_images_paths)
            
            target_class_indices = self.class_names2indices(cnk.target_class_name)
            if np.isnan(g.interp_alpha):
                im_adv, budget_usage = attacker_model.attack_target(images_source, target_class_indices, 
                                                                    eps=budget, 
                                                                    step_size=step_size, 
                                                                    n_iter=g.n_iter, 
                                                                    constraint=g.constraint,
                                                                    do_tqdm=True)
                            
            else:  # Contrast blend
                target_images_paths = [self.rng_job.choice(self.data_dict[class_name]) for class_name in cnk.target_class_name]
                images_target = self.get_images(target_images_paths)
                im_adv, budget_usage = contrast_blend(images_source, images_target, g.interp_alpha, eps=budget, constraint=g.constraint)
                for image_id, img_target_path in zip(cnk.image_id, target_images_paths):
                    ds.target_image_path.loc[dict(image_id=image_id)] = img_target_path[len(f"{self.data_root}/"):]
                    
            
            for image_id, img_source_path, img_pil, img_budget_usage in zip(cnk.image_id, 
                                                                            source_images_paths,
                                                                            tor2pil(im_adv), 
                                                                            budget_usage.float(), 
                                                                            ):
                os.makedirs(f"{self.output_folder}/images", exist_ok=True)
                Image.fromarray(img_pil).save(f"{self.output_folder}/images/{image_id}")
                ds.orig_name.loc[dict(image_id=image_id)] = img_source_path[len(f"{self.data_root}/"):]
                ds.budget_usage.loc[dict(image_id=image_id)] = np.nan_to_num(img_budget_usage) 
        self.model_subjects_predict(ds, df_agg, g, batch_size)
        ds.sel(image_id=list(df_agg.image_id)).to_netcdf(f"{self.output_folder}/{meta_filename}")
