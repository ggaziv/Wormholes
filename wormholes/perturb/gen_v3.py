"""
    Implementation of RestrictedImageNet-based category modulation by attacking model logits
    Like V1 but for demo: For each source image, generate attacks toward all 8 possible targets, given all possible models, across all budgets.
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v1 import GenV1


class GenV3(GenV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        args = self.args  
         
        self.rng_job = np.random.default_rng(int(args.seed or 0))
              
        configs = list(itertools.product(self.model_kwargs_dict.items(), self.attack_hparams_tup_list)) + self.interp_hparams_tup_list
        cprintm(f"Total number of configs: {len(configs)}")
        
        ds = self.get_ds()
        df = self.dataset_to_dataframe(ds)
        
        # Screen for current job
        group, df_agg = self.screen_job_work(df)
        
        self.make_and_save_results(ds, df_agg, group, 
                                   batch_size=args.batch_size)
    
    def get_data(self, n_sample_class=1):
        triplet_paths_list = []
        for i, (class_name, image_paths) in enumerate(self.data_dict.items()):
            for j in range(n_sample_class):
                seed = i * n_sample_class + j
                img_src_tup = (np.random.RandomState(seed).choice(image_paths)[len(f"{self.data_root}/"):], class_name)
                classes_other = [c for c in self.data_dict if c != class_name]
                triplet_paths_list.extend([[img_src_tup, target_class_name] for target_class_name in classes_other])
        return triplet_paths_list
    
    def get_gen(self, hparams_tup_list, triplet_paths_list):
        for hp in hparams_tup_list:
            for triplet_index, triplet_paths in enumerate(triplet_paths_list):
                image, class_name = triplet_paths[0]
                assert len(triplet_paths) == 2
                # The distract class is the same of the source
                yield hp, triplet_index, image, triplet_paths[1], class_name
                    
    def add_contrast_blend(self, meta_list, gen):
        for hp, triplet_index, orig_path, target_class_name, orig_class_name in gen(self.interp_hparams_tup_list):
            row = [
                'contrast_blend', 
                hp.eps, 0, np.nan, hp.alpha_interp,
                target_class_name,
                orig_class_name,
                orig_path,
                triplet_index,
                ]
            meta_list.append(row + [np.nan] * (len(self.field_dict) - len(row)))
                         
    def add_attack(self, meta_list, gen):
        for model_name in self.model_kwargs_dict:
            for hp, triplet_index, orig_path, target_class_name, orig_class_name in gen(self.attack_hparams_tup_list):
                row = [
                    model_name, 
                    hp.eps, hp.n_iter, hp.step_size, np.nan,
                    target_class_name,
                    orig_class_name,
                    orig_path,
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
            images_source = self.get_images(list(cnk.orig_name.apply(lambda s: f"{self.data_root}/{s}")))
            target_class_indices = self.class_names2indices(cnk.target_class_name)
            im_adv, budget_usage = self.make_adv(ds, g, attacker_model, cnk, images_source, target_class_indices)

            for image_id, img_pil, img_budget_usage in zip(cnk.image_id, 
                                                           tor2pil(im_adv), 
                                                           budget_usage.float(), 
                                                           ):
                os.makedirs(f"{self.output_folder}/images", exist_ok=True)
                Image.fromarray(img_pil).save(f"{self.output_folder}/images/{image_id}")
                ds.budget_usage.loc[dict(image_id=image_id)] = np.nan_to_num(img_budget_usage) 
        self.model_subjects_predict(ds, df_agg, g, batch_size)
        ds.sel(image_id=list(df_agg.image_id)).to_netcdf(f"{self.output_folder}/{meta_filename}")
        