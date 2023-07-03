"""
    Implementation of RestrictedImageNet-based category modulation by attacking model logits
"""
    
    
from wormholes.perturb.utils import *


class GenV6:
    field_dict = {
        'model_name': str,
        'budget': float, 'n_iter': int, 'step_size': float, 'interp_alpha': float,
        'target_class_name': str,
        'orig_class_name': str,
        'orig_name': str,
        'triplet': int,
        'budget_usage': float, 
        'target_image_path': str,
        }
    group_columns = ['model_name', 'budget', 'n_iter', 'step_size', 'interp_alpha']
    class_dict = CLASS_DICT['RestrictedImageNet']
    attack_hparams_tup_list = [namedtuple('attack_hparams', ['eps', 'step_size', 'n_iter'])(*x) 
                               for x in [(300, 4, 10_000), (200, 3, 10_000), (150, 2, 10_000), 
                                         (100, 2, 5000), (50, 2, 2000), (30, .5, 1000), 
                                         (10, .5, 500), (5, .5, 500), 
                                         (2.5, .3, 200), (1., .3, 200), (.5, .1, 200), (.1, .02, 200),
                                         (0., 0., 0)
                                         ]]
    interp_hparams_tup_list = [namedtuple('interp_hparams', ['eps', 'alpha_interp'])(*x) 
                               for x in itertools.product([300, 200, 150, 100, 50, 30, 10, 5, 2.5, 1., .5, .1], 
                                                          [0.])]
    
    def __init__(self, args=None, data_root=f"{PROJECT_ROOT}/data"):
        seed = 0 if args is None else args.seed
        set_seed(seed)
        self.gen_version = extract_from_string(str(self.__class__), 'GenV')
        self.args = args
        self.data_root = data_root
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
                        }
                    } for eps in [1, 3, 10]
            }
        # Add the self-subject to all attackers
        for model_name, d in self.model_kwargs_dict.items():
            d['model_subjects'][model_name] = d['attacker']
            
        self.contrast_blend_model_subjects = {
            'resnet50_vanilla_mapped_RIN_v2': partial(get_mapped_RIN, 'imagenet_vanilla_v2.pt')
            } | {
                f'resnet50_robust_mapped_RIN_l2_{eps_eval}_0_v2': partial(get_mapped_RIN, f'imagenet_l2_{eps_eval}_0_v2.pt') 
                for eps_eval in [1, 3, 10]    
                }

        folder_ds = ImageFolder(root=f"{ds.data_path}/val", label_mapping=ds.label_mapping)
        data_dict = invert_dict(dict(folder_ds.samples))
        k = min([len(v) for v in data_dict.values()])
        self.data_dict = {self.class_dict[int(lbl)]: v[:k] for lbl, v in data_dict.items()}
        self.class_name_to_index = {v: k for k, v in self.class_dict.items()}
        
        # This includes an anti-aliasing resize on a PIL image
        self.transform = transforms.Compose([transforms.ToPILImage()] + 
                                            ds.transform_test.transforms[:2] + 
                                            [np.array])
        
        self.model_subjects_names = np.unique(chained(list(x['model_subjects'].keys()) for x in self.model_kwargs_dict.values()))
        self.output_folder = f"{PROJECT_ROOT}/results/cache/gen_v{self.gen_version}"
        
    def run(self):
        args = self.args
        self.rng_job = np.random.default_rng(int(args.num or 0))
        
        configs = list(itertools.product(self.model_kwargs_dict.items(), self.attack_hparams_tup_list)) + self.interp_hparams_tup_list
        cprintm(f"Total number of configs: {len(configs)}")
        
        ds = self.get_ds()
        df = self.dataset_to_dataframe(ds)
        
        # Screen for current job
        group, df_agg = self.screen_job_work(df)
        self.make_and_save_results(ds, df_agg, group, 
                                   batch_size=args.batch_size)
        
    def screen_job_work(self, df, print=True):
        columns_group = self.group_columns
        if print:
            for g_i, g in enumerate(df.groupby(columns_group).groups):
                cprint1(f"{g_i}) {g}")
        group, df_indices = list(df.groupby(columns_group).groups.items())[self.args.num]
        group = namedtuple('group', columns_group)(*group)
        cprint1(f"(*) {group}")
        df_agg = df.loc[df_indices]
        return group, df_agg

    def get_ds(self):
        triplet_paths_list = self.get_data()
        ds = self.make_meta_ds(triplet_paths_list)
        
        # Add model-subjects prediction fields
        make_da = lambda : xr.DataArray(np.full((len(ds.image_id), len(self.model_subjects_names)), np.nan), 
                                                 coords=dict(
                                                     image_id=ds.image_id, 
                                                     model_subject_name=self.model_subjects_names))
        ds = ds.assign(
            pred_logit=xr.DataArray(np.full((len(ds.image_id), len(self.data_dict), len(self.model_subjects_names)), np.nan), 
                                    coords=dict(
                                        image_id=ds.image_id, 
                                        class_index=np.arange(len(self.data_dict)), 
                                        model_subject_name=self.model_subjects_names)),
            pred_prob_choose_target=make_da(),
            pred_prob_choose_target_pairwise_softmax=make_da(),
            pred_prob_choose_target_softmax=make_da(),
        )
        for k in ['orig_name', 'target_image_path']:
            ds[k] = ds[k].astype('object')
        return ds
    
    def make_meta_ds(self, triplet_paths_list):
        meta_list = self.make_meta_list(triplet_paths_list)
        assert len(meta_list), "No data found."
        da = xr.DataArray(meta_list)
        ds = da.to_dataset('dim_1').rename_vars({i: name for i, name in enumerate(self.field_dict)})
        for k, caster in self.field_dict.items():
            ds[k] = ds[k].astype(caster)
        ds = ds.rename({'dim_0': 'image_id'})
        rd = random.Random()
        rd.seed(self.gen_version)
        ds = ds.assign_coords(image_id=[f"{uuid.UUID(int=rd.getrandbits(128)).hex}.png" for _ in ds.image_id])
        return ds
    
    def make_meta_list(self, triplet_paths_list):
        meta_list = []
        gen = lambda hparams_tup_list: self.get_gen(hparams_tup_list, triplet_paths_list)
        self.add_attack(meta_list, gen)
        # Add contrast-blend controls
        self.add_contrast_blend(meta_list, gen)
        return meta_list

    def get_data(self, n_sample_class=3):
        triplet_paths_list = []
        for i, (class_name, image_paths) in enumerate(self.data_dict.items()):
            for j in range(n_sample_class):
                classes_other = [c for c in self.data_dict if c != class_name]
                triplet_paths_list.extend([[class_name, target_class_name] for target_class_name in classes_other])
        return triplet_paths_list
    
    def get_gen(self, hparams_tup_list, triplet_paths_list):
        for hp in hparams_tup_list:
            for triplet_index, triplet_paths in enumerate(triplet_paths_list):
                class_name = triplet_paths[0]
                assert len(triplet_paths) == 2
                # The distract class is the same of the source
                yield hp, triplet_index, triplet_paths[1], class_name
    
    def add_contrast_blend(self, meta_list, gen):
        for hp, triplet_index, target_class_name, orig_class_name in gen(self.interp_hparams_tup_list):
            row = [
                'contrast_blend', 
                hp.eps, 0, np.nan, hp.alpha_interp,
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
                    hp.eps, hp.n_iter, hp.step_size, np.nan,
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
        else:
            attacker_model = None
        for cnk_i, cnk in enumerate(df_chunks):
            cprint1(f"chunk [{cnk_i+1}/{len(df_chunks)}]")
            source_images_paths = [self.rng_job.choice(self.data_dict[class_name]) for class_name in cnk.orig_class_name]
            images_source = self.get_images(source_images_paths)
            
            target_class_indices = self.class_names2indices(cnk.target_class_name)
            im_adv, budget_usage = self.make_adv(ds, g, attacker_model, cnk, images_source, target_class_indices)
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

    def make_adv(self, ds, g, attacker_model, cnk, images_source, targets):
        if np.isnan(g.interp_alpha):
            im_adv, budget_usage = attacker_model.attack_target(images_source, targets, 
                                                                eps=g.budget, 
                                                                step_size=g.step_size, 
                                                                n_iter=g.n_iter, 
                                                                do_tqdm=True)
                            
        else:  # Contrast blend
            target_images_paths = [self.rng_job.choice(self.data_dict[class_name]) for class_name in cnk.target_class_name]
            images_target = self.get_images(target_images_paths)
            im_adv, budget_usage = contrast_blend(images_source, images_target, g.interp_alpha, eps=g.budget)
            for image_id, img_target_path in zip(cnk.image_id, target_images_paths):
                ds.target_image_path.loc[dict(image_id=image_id)] = img_target_path[len(f"{self.data_root}/"):]
        return im_adv, budget_usage
    
    def class_names2indices(self, pd_series):
        return pd_series.apply(lambda s: self.class_name_to_index[s]).to_numpy(dtype=int)
    
    def model_subjects_predict(self, ds, df_agg, g, batch_size=50):
        from torch.nn.functional import softmax
        if np.isnan(g.interp_alpha):
            model_subjects_dict = self.model_kwargs_dict[g.model_name]['model_subjects']
        else:
            model_subjects_dict = self.contrast_blend_model_subjects
        
        for model_subject_name, model_subject_maker in model_subjects_dict.items():
            cprint1(f"Model-subject [{model_subject_name}]")
            model_subject = model_subject_maker().eval()
            with torch.no_grad():
                for cnk in tqdm(chunks(df_agg, batch_size)):
                    im_adv_paths = cnk.image_id.apply(lambda s: f"{self.output_folder}/images/{s}")
                    im_adv = pil2tor(self.get_images(im_adv_paths, use_transform=False))
                    pred = model_subject.predict(im_adv, return_numpy=False)
                    for image_id, logits, orig_class_idx, target_class_idx in zip(cnk.image_id, pred.float(),
                                                                           self.class_names2indices(cnk.orig_class_name), 
                                                                           self.class_names2indices(cnk.target_class_name)):
                        index_dict = dict(image_id=image_id, model_subject_name=model_subject_name)
                        ds.pred_logit.loc[index_dict] = logits
                        ds.pred_prob_choose_target.loc[index_dict] = logits[target_class_idx] > logits[orig_class_idx]
                        ds.pred_prob_choose_target_softmax.loc[index_dict] = softmax(logits)[target_class_idx]
    
    def get_images(self, *args, use_transform=True):
        images = get_images(*args)
        if use_transform:
            return np.stack(list(map(self.transform, images)))
        else:
            return np.array(images)
        
    def make_attacker_model(self, model_name):
        if model_name == 'contrast_blend':
            model = types.SimpleNamespace()
        else:
            model = self.model_kwargs_dict[model_name]['attacker']()
            model.eval()
        model.attack_target = types.MethodType(attack_target_class, model)
        return model        
    
    @staticmethod
    def dataset_to_dataframe(ds):
        ds1 = ds.drop_dims(dim_name for dim_name in ds.dims if dim_name != 'image_id')
        assert len(ds1.dims) == 1
        return ds1.to_dataframe().reset_index()