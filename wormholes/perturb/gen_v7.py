"""
    Like V6 but with AnimalImageDataset (Kaggle) targets based on a centroid-approach.
    https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v6 import GenV6


class GenV7(GenV6):   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Drop robust model 1.0
        self.model_kwargs_dict = {k: v for k, v in self.model_kwargs_dict.items() if k not in ['resnet50_robust_mapped_RIN_l2_1_0']}
        self.contrast_blend_model_subjects = {k: v for k, v in self.contrast_blend_model_subjects.items() if k not in ['resnet50_robust_mapped_RIN_l2_1_0_v2']}
        self.model_subjects_names = np.unique(chained(list(x['model_subjects'].keys()) for x in self.model_kwargs_dict.values()))
        
        self.data_dict_OOD = {class_name: glob.glob(f"{self.data_root}/OOD/{class_name}/*") for class_name in self.data_dict}
        self.data_ANI = glob.glob(f"{self.data_root}/ANI/*")
        
        self.attack_hparams_tup_list = [namedtuple('attack_hparams', ['eps', 'step_size', 'n_iter'])(*x) 
                                        for x in [(50, 2, 2000), (40, 2, 2000), (30, 2, 2000), (25, .5, 1000), (20, .5, 1000), 
                                                  (0., 0., 0)
                                                  ]]
        
        self.interp_hparams_tup_list = [namedtuple('interp_hparams', ['eps', 'alpha_interp'])(*x) 
                               for x in itertools.product([50, 40, 30, 25, 20], 
                                                          [0.])]
        
        # Target classes by image folders (centroid approach)  
        self.class_paths = glob.glob(f"{self.data_root}/AnimalImageDataset/custom_v0/*")   
        
    def report_target_class_image_counts(self):
        d = {os.path.basename(class_path): len(os.listdir(class_path)) for class_path in self.class_paths}
        cprintm(f"Image counts:\n{d}")
    
    def get_target_class_dict(self):
        return {i: class_path[len(f"{self.data_root}/"):] for i, class_path in enumerate(self.class_paths)}
    
    def run(self):
        args = self.args  
        self.report_target_class_image_counts()
        self.class_name_to_index = {class_path[len(f"{self.data_root}/"):]: i for i, class_path in enumerate(self.class_paths)}
         
        # self.rng_job = np.random.default_rng(int(args.num or 0))
        self.rng_job = np.random.default_rng(int(args.seed or 0))
              
        configs = list(itertools.product(self.model_kwargs_dict.items(), self.attack_hparams_tup_list)) + self.interp_hparams_tup_list
        cprintm(f"Total number of configs: {len(configs)}")
        
        ds = self.get_ds()
        df = self.__class__.dataset_to_dataframe(ds)
        
        # Screen for current job
        group, df_agg = self.screen_job_work(df)
        
        self.make_and_save_results(ds, df_agg, group, 
                                   batch_size=args.batch_size)
        
    def get_data(self):
        prefix = f"{self.data_root}/"
        
        # Arbitrary Natural Images
        triplet_paths_list = [[(img_path[len(prefix):], 'ANI'), class_path[len(prefix):]] 
                              for img_path in np.random.choice(self.data_ANI, size=20, replace=False) 
                              for class_path in self.class_paths]
        # Out of distribution from RestrictedImageNet classes
        for class_name in self.data_dict:
            triplet_paths_list += [[(img_path[len(prefix):], f'OOD-{class_name}'), class_path[len(prefix):]] 
                                   for img_path in np.random.choice(self.data_dict_OOD[class_name], size=10, replace=False)
                                   for class_path in self.class_paths if class_path[len(prefix):] != class_name]
        # Uniform Noise Image
        triplet_paths_list += [[('', 'UNI'), class_path[len(prefix):]] for class_path in self.class_paths] * 5
        return triplet_paths_list
                                       
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
            images_source = self.get_images(list(cnk.orig_name.apply(lambda s: f"{self.data_root}/{s}")))
            images_target = self.get_images_dynamic(list(cnk.target_class_name.apply(lambda s: f"{self.data_root}/{s}")), k_images_class=batch_size)
            im_adv, budget_usage = self.make_adv(ds, g, attacker_model, cnk, images_source, images_target)

            for image_id, img_pil, img_budget_usage in zip(cnk.image_id, 
                                                           tor2pil(im_adv), 
                                                           budget_usage.float(), 
                                                           ):
                os.makedirs(f"{self.output_folder}/images", exist_ok=True)
                Image.fromarray(img_pil).save(f"{self.output_folder}/images/{image_id}")
                ds.budget_usage.loc[dict(image_id=image_id)] = np.nan_to_num(img_budget_usage) 
        self.model_subjects_predict(ds, df_agg, g, batch_size)
        ds.sel(image_id=list(df_agg.image_id)).to_netcdf(f"{self.output_folder}/{meta_filename}")

    def get_images_dynamic(self, file_paths_list, k_images_class=100, **kwargs):
        dirs_only, files_only = [all(x) for x in zip(*[(os.path.isdir(fp), os.path.isfile(fp)) for fp in file_paths_list])]
        assert dirs_only or files_only, f"Received a mix of files and folders {file_paths_list}."
        if dirs_only:
            return [self.get_images(natsorted(glob.glob(f"{fp}/*"))[:k_images_class], **kwargs) for fp in file_paths_list]
        else:
            return self.get_images(file_paths_list, **kwargs)
        
    def make_adv(self, ds, g, attacker_model, cnk, images_source, targets):
        if np.isnan(g.interp_alpha):
            im_adv, budget_usage = attacker_model.attack_target(images_source, targets, 
                                                                eps=g.budget, 
                                                                step_size=g.step_size, 
                                                                n_iter=g.n_iter, 
                                                                do_tqdm=True)
                            
        else:  # Contrast blend
            target_images_paths = [self.rng_job.choice(glob.glob(f"{self.data_root}/{class_name}/*")) for class_name in cnk.target_class_name]
            images_target = self.get_images(target_images_paths)
            im_adv, budget_usage = contrast_blend(images_source, images_target, g.interp_alpha, eps=g.budget)
            for image_id, img_target_path in zip(cnk.image_id, target_images_paths):
                ds.target_image_path.loc[dict(image_id=image_id)] = img_target_path[len(f"{self.data_root}/"):]
        return im_adv, budget_usage
    
    def make_attacker_model(self, model_name):
        if model_name == 'contrast_blend':
            model = types.SimpleNamespace()
        else:
            model = self.model_kwargs_dict[model_name]['attacker']()
            model.eval()
            attacker = model.attacker
            embedder = lambda inp: model.model.forward(inp, with_latent=True)[-1]
            attacker.attack_target = types.MethodType(
                partial(attack_target_centroid, 
                        embedder=embedder, 
                        criterion=mse_batchwise), attacker)
        return attacker        