"""
    Like V3 (demo V1) but with the same image-target pairs across model and budgets, a narrower budget regime, and start images of three types 
    (i) Arbitrary natural images (not from 9 categories) (ANI)
    (ii) Natural images (from 9 categories) -- not from Imagenet Val (OOD)
    (iii) Random starts (uniform noise) (UNI)
"""
    
    
from wormholes.perturb.utils import *
from wormholes.perturb.gen_v3 import GenV3


class GenV6(GenV3):
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
        
    def get_data(self):
        # Arbitrary Natural Images
        triplet_paths_list = [[(img_path[len(f"{self.data_root}/"):], 'ANI'), target_class_name] 
                              for img_path in np.random.choice(self.data_ANI, size=20, replace=False) 
                              for target_class_name in self.data_dict]
        # Out of distribution from RestrictedImageNet classes
        for class_name in self.data_dict:
            triplet_paths_list += [[(img_path[len(f"{self.data_root}/"):], f'OOD-{class_name}'), target_class_name] 
                                   for img_path in np.random.choice(self.data_dict_OOD[class_name], size=10, replace=False)
                                   for target_class_name in self.data_dict if target_class_name != class_name]
        # Uniform Noise Image
        triplet_paths_list += [[('', 'UNI'), target_class_name] for target_class_name in self.data_dict] * 5
        return triplet_paths_list
    
    def get_images(self, image_paths, use_transform=True):
        empty_path = f"{self.data_root}/"
        image_paths1 = [img_path for img_path in image_paths if img_path != empty_path]
        if len(image_paths1):
            source_images1 = list(super().get_images(image_paths1, use_transform=use_transform))
        else:
            source_images1 = []
        img_list = []
        for img_path in image_paths:
            if img_path == empty_path:
                img_list.append((self.rng_job.random((im_res, im_res, 3)) * 255).astype('uint8'))
            else:
                img_list.append(source_images1.pop(0))
        assert len(source_images1) == 0
        return np.stack(img_list)
    
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
                    for image_id, logits, target_class_idx in zip(cnk.image_id, pred.float(),
                                                                  self.class_names2indices(cnk.target_class_name)):
                        index_dict = dict(image_id=image_id, model_subject_name=model_subject_name)
                        ds.pred_logit.loc[index_dict] = logits
                        ds.pred_prob_choose_target_softmax.loc[index_dict] = softmax(logits)[target_class_idx]