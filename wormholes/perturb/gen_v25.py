"""
    Non-cardinal attacks on RestrictedImageNet
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v6 import GenV6


class GenV25(GenV6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_kwargs_dict = {k: v for k, v in self.model_kwargs_dict.items() if k in ['resnet50_robust_mapped_RIN_l2_3_0', 
                                                                                           'resnet50_robust_mapped_RIN_l2_10_0']}
        self.contrast_blend_model_subjects = {k: v for k, v in self.contrast_blend_model_subjects.items() if k in ['resnet50_robust_mapped_RIN_l2_3_0_v2',
                                                                                                                   'resnet50_robust_mapped_RIN_l2_10_0_v2']}
        self.model_subjects_names = np.unique(chained(list(x['model_subjects'].keys()) for x in self.model_kwargs_dict.values()))
        class_dict = {k: v for k, v in self.class_dict.items() if k >= 0}
        self.targets_dict = (
            AttackMultiTargetUniform.make_comb_targets(class_dict, k_comb=2, n_sample_combs=4) |
            AttackMultiTargetUniform.make_comb_targets(class_dict, k_comb=3, n_sample_combs=3)
            )
        
    def get_data(self, n_sample_class=30):
        triplet_paths_list = []
        for class_name in self.data_dict:
            for j in range(n_sample_class):
                classes_other = [c for c in self.targets_dict if c != class_name]
                triplet_paths_list.extend([[class_name, target_class_name] for target_class_name in classes_other])
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
            source_images_paths = [self.rng_job.choice(self.data_dict[class_name]) for class_name in cnk.orig_class_name]
            images_source = self.get_images(source_images_paths)
            
            target_logits = self.class_names2target_logits(cnk.target_class_name)
            im_adv, budget_usage = self.make_adv(ds, g, attacker_model, cnk, images_source, target_logits)

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

    def make_adv(self, ds, g, attacker_model, cnk, images_source, target_logits):
        if np.isnan(g.interp_alpha):
            im_adv, budget_usage = attacker_model.attack_target(images_source, target_logits, 
                                                                custom_loss=self.custom_loss,
                                                                eps=g.budget, 
                                                                step_size=g.step_size, 
                                                                n_iter=g.n_iter, 
                                                                do_tqdm=True)
                            
        else:  # Contrast blend
            target_images_paths_list = [[self.rng_job.choice(self.data_dict[cn]) for cn in self.targets_dict[class_name]['compos_class_names']] 
                                   for class_name in cnk.target_class_name]
            # Create uniformly interpolated target image
            images_target = np.stack([self.get_images(target_images_paths).mean(0) for target_images_paths in target_images_paths_list])
            # For "content removal" case, override with random pixels as targets
            ind_bool = [class_name == 'none' for class_name in cnk.target_class_name]
            images_target[ind_bool] = (self.rng_job.random((sum(ind_bool), im_res, im_res, 3)) * 255).astype('uint8')
            im_adv, budget_usage = contrast_blend(images_source, images_target, g.interp_alpha, eps=g.budget)
        return im_adv, budget_usage
    
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
                    for image_id, logits, orig_class_idx in zip(cnk.image_id, pred.float(),
                                                                self.class_names2indices(cnk.orig_class_name)):
                        index_dict = dict(image_id=image_id, model_subject_name=model_subject_name)
                        ds.pred_logit.loc[index_dict] = logits
    
    def class_names2target_logits(self, pd_series):
        target_logits = np.array(np.stack([self.targets_dict[s]['target_logits'] for s in list(pd_series)]))
        target_logits[target_logits < 0] = 0
        target_probs = target_logits / target_logits.sum()
        return target_probs
    
    def make_attacker_model(self, model_name):
        model = self.model_kwargs_dict[model_name]['attacker']()
        model.eval()
        model.attack_target = types.MethodType(attack_composite_target, model)
        return model    
    
    @staticmethod
    def custom_loss(model, inp, target_probs, high_target=1000):
        pred = model(inp)
        loss = cross_entropy(pred, target_probs, reduction='none')
        pred_gather = torch.stack([pred_row[target_probs_row > 0].mean() for pred_row, target_probs_row in zip(pred, target_probs)])
        loss += (pred_gather - high_target)**2 / high_target**2
        return loss, None
    