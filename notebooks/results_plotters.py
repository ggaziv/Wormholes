from results_analysis_utils import *


BRIGHT_COLORS = dict(zip(['blue', 'orange', 'green', 'red', 'magenta', 'brown', 'pink', 'gray', 'yellow', 'cyan'], 
                         sns.color_palette('bright', 10)))
DARK_RED = '#BE1E2D'


class Plotter:
    def __init__(self, gen_data_name, behavior_data_name=None, x_var='budget', 
                 lapse_rate_norm=True, class_dict=None,
                 worker_mean=True):
        if class_dict is None:
            self.class_dict = dict(list(CLASS_DICT['RestrictedImageNet'].items())[1:])
        else: 
            self.class_dict = class_dict
        self.x_var = x_var
        self.lapse_rate_norm = lapse_rate_norm
        self.worker_mean = worker_mean
        ds = xr.open_dataset(f"{PROJECT_ROOT}/results/cache/{gen_data_name}")
        # Cast some some fields to string
        for field in ['orig_name', 'orig_class_name', 'target_class_name']:
            ds[field] = ds[field].astype(str)
        ds = self.adjust_from_behavior(ds, behavior_data_name)
        ds = self.add_decision_rules(ds)
        ds = self.compute_derivarives(ds)
        
        if behavior_data_name is None:
            dim_name = list(ds.dims)[0]
            ds['mean_prob_choose_target'] = xr.DataArray(np.full(len(ds[dim_name]), np.nan), dims=dim_name)
            ds['mean_prob_choose'] = xr.full_like(ds.pred_logit, np.nan)
        else:
            ds = self.add_behavior_data(ds, behavior_data_name)
        
        ds = self.discard_surplus_controls(ds)
        
        ds = self.arrange_case(ds)
        self.ds = ds
    
    def discard_surplus_controls(self, ds):
        """Retain all control data from just the first model and discard the rest
        """
        model_names_with_controls = np.unique(ds.where(ds.budget == 0, drop=True).model_name.values)
        ds = ds.where((ds.budget > 0) | (ds.model_name == model_names_with_controls[0]), drop=True).copy()
        return ds

    def arrange_case(self, ds):
        ds['case'] = ds.model_name.copy()
        ds['case'][ds.model_name == 'contrast_blend'] = 'Contrast-blend'
        ds['case'][ds.budget == 0] = 'No perturb.'
        return ds
    
    def adjust_from_behavior(self, ds, behavior_data_name):
        return ds
    
    def add_decision_rules(self, ds):
        return ds
    
    def compute_derivarives(self, ds):
        return ds
    
    def adjust_ylabel(self, ylabel):
        if self.lapse_rate_norm:
            return ('Normalized ' + ylabel).capitalize()
        else:
            return ylabel
    
    def plot_Model_Human_Alignment(self,
                                   pred_field='pred_prob_choose_target_v2_ts0.01_cal',
                                   attacker_type_only=True,
                                   ylabel="Prob. choose target",
                                   bbox_to_anchor=(1.5, .8),
                                   custom_palette=None):
        df = self.get_df_Model_Human_Alignment(pred_field, attacker_type_only)
        obj = Model_Human_Alignment(df,
                            x=self.x_var,
                            model_var=self.make_model_var(self.ds),
                            add_inferred_catch=self.add_inferred_catch,
                            )
        return obj.plot(concise=False, ylabel=self.adjust_ylabel(ylabel), bbox_to_anchor=bbox_to_anchor, 
                        custom_palette=custom_palette)

    def plot_Alignment(self,
                       pred_field='pred_prob_choose_target',
                       attacker_type_only=False,
                       ylabel="Prob. choose target",
                       add_inferred_catch=True,
                       bbox_to_anchor=(1.5, .8),
                       custom_palette=None):
        df = self.get_df_Model_Human_Alignment(pred_field, attacker_type_only)
        obj = AlignmentPlot(df,
                            x=self.x_var,
                            model_var=self.make_model_var(self.ds),
                            add_inferred_catch=self.add_inferred_catch,
                            )
        obj.plot(concise=False, ylabel=self.adjust_ylabel(ylabel), bbox_to_anchor=bbox_to_anchor,
                 custom_palette=custom_palette)
    
    def plot_Human_Model_2AFC(self, pred_field='pred_prob_choose_target_v2_ts0.01_cal'):
        df = self.get_df_Human_Model_2AFC(pred_field)
        Human_Model_2AFC(df).plot(
            x=self.x_var,
            var_list={var: self.adjust_ylabel(var) for var in [
                'mean_prob_choose_target',
                pred_field,
                ]},
            add_inferred_catch=self.add_inferred_catch)
        
    @staticmethod
    def make_model_var(ds):
        return {name: ' '.join(name.replace('_0', '').replace('_mapped','').replace('_l2', '').split('_')[1:]) 
                                       for name in ds.model_subject_name.values}
        
    @staticmethod
    def get_collapsed_worker_id(behavior_data_name):
        ds1 = xr.open_dataset(f"{PROJECT_ROOT}/results/behavior/{behavior_data_name}")
        cprintm("Collapsing workers as new coordinate...")
        ds1['stimulus_id'] = ds1.stimulus_id.astype(str)
        ds1 = xr.concat([ds_worker.groupby('stimulus_id').sum().assign_coords(worker_id=worker_id) 
                         for worker_id, ds_worker in tqdm(ds1.groupby('worker_id'))], 'worker_id')
        return ds1


###########################################################
################## 9-WAY TARGETED ATTACKS #################
###########################################################

    
class Plotter_9Way(Plotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_inferred_catch = False
        
        
    def compute_derivarives(self, ds, 
                            temp=1
                            ):
        # Change class_index => class_name
        ds = ds.assign_coords(class_index=[self.class_dict[i] for i in ds.class_index.values]).rename(class_index='class_name')
        # Compute predicted probability
        ds['pred_prob_choose'] = scipy.special.softmax(ds.pred_logit / temp, axis=list(ds.dims).index('class_name'))
        return ds
    
    def get_ds_Model_Human_Alignment_MultiWay(self, pred_field, attacker_type_only):
        ds = self.ds
        ds1 = ds.copy()
        if attacker_type_only:
            # Include as subject models only models from attacker type (including attacker)
            ds1[pred_field] = ds1[pred_field].where(((ds1.model_subject_name == ds1.model_name) | 
                                                    (ds1.model_subject_name == ds1.model_name + '_v2') | 
                                                    (ds1.model_name == 'contrast_blend')))

        ds1 = ds1.drop_dims('model_subject_name').merge(ds1[pred_field].to_dataset('model_subject_name'))
        return ds1
    
    def get_df_Model_Human_Alignment_MultiWay(self, 
                                              pred_field='pred_prob_choose',
                                              attacker_type_only=True):
        ds1 = self.get_ds_Model_Human_Alignment_MultiWay(pred_field, attacker_type_only)
        df = ds1.to_dataframe().reset_index()
        return df

    def get_df_Model_Human_Alignment(self, 
                                    pred_field='pred_prob_choose',
                                    attacker_type_only=True):
        ds1 = self.collapse_to_1d(pred_field, self.ds)
        pred_field = pred_field.replace('_choose', '_choose_target')
        ds1 = self.get_ds_Model_Human_Alignment_MultiWay(pred_field, attacker_type_only)
        
        df = ds1.to_dataframe().reset_index()
        return df
    
    def get_df_Human_Model_2AFC(self, pred_field='pred_prob_choose'):
        ds = self.ds
        ds = self.collapse_to_1d(pred_field, ds)
        # Use self-subject model v2
        a = ds.model_name.copy()
        a.loc[a=='contrast_blend'] = 'resnet50_vanilla_mapped_RIN'
        
        ds_list = [ds.sel(model_subject_name=a + '_v2').drop_dims(['worker_id'], errors='ignore')]
        # Behavioral data might have worker_id. Avoid replicating other data_vars 
        if 'worker_id' in ds.coords:
            ds_list.append(ds.drop_dims(['model_subject_name']))
        df = pd.concat([x.to_dataframe().reset_index() for x in ds_list]).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def collapse_to_1d(pred_field, ds1):
        ds1[pred_field.replace('_choose', '_choose_target')] = ds1[pred_field].sel(class_name=ds1.target_class_name.astype(str))
        if 'mean_prob_choose' in ds1.data_vars:
            ds1['mean_prob_choose_target'] = ds1['mean_prob_choose'].sel(class_name=ds1.target_class_name)
        ds1 = ds1.drop_dims('class_name')
        return ds1


class GenV5_Plotter_9Way(Plotter_9Way):      
    def add_behavior_data(self, ds, behavior_data_name):
        ds1 = xr.open_dataset(f"{PROJECT_ROOT}/results/behavior/{behavior_data_name}")
        if 'worker_id' not in ds1.k.dims:
            strs = behavior_data_name.split('/')
            behavior_data_name1 = '/'.join(strs[:-1] + [strs[-1].replace('flat', 'flat_collapsed_workers')])
            behavior_data_path1 = f"{PROJECT_ROOT}/results/behavior/{behavior_data_name1}"
            force_override = False
            if os.path.exists(behavior_data_path1) and not force_override:
                cprintm(f"Loading collapsed file: {behavior_data_name1}")
                ds1 = xr.open_dataset(behavior_data_path1)
            else:
                ds1 = self.get_collapsed_worker_id(behavior_data_name)
                cprintm(f"Saving {behavior_data_name1}")
                ds1.to_netcdf(behavior_data_path1)
        
        ds1 = self.setup_behavior_data(ds1, ds)
            
        ds['n_selected'] = xr.DataArray(np.zeros((len(ds1.worker_id), len(ds.image_id), len(ds.class_name))), 
                                        coords=dict(
                                            worker_id=ds1.worker_id,
                                            image_id=ds.image_id, 
                                            class_name=ds.class_name,
                                        ))
        ds['n_reps'] = xr.DataArray(np.zeros((len(ds1.worker_id), len(ds.image_id))),
                                    coords=dict(
                                        worker_id=ds1.worker_id,
                                        image_id=ds.image_id, 
                                    ))
        
        ds['n_selected'].loc[dict(image_id=ds1.stimulus_id)] = ds1.k.sel(choice=ds.class_name)
        ds['n_reps'].loc[dict(image_id=ds1.stimulus_id)] = ds1.n
        
        ds['mean_prob_choose'] = ds.n_selected / ds.n_reps
        if self.lapse_rate_norm:
            ds_calib = xr.open_dataset(f"{PROJECT_ROOT}/results/behavior/{behavior_data_name.replace('flat', 'calibration')}")
            catch_perfs = ds_calib.kcatch / ds_calib.ncatch
            gamma = LapseRateNorm.compute_gamma(catch_perfs, N=9)
            ds['mean_prob_choose'] = LapseRateNorm.apply_gamma(ds['mean_prob_choose'], gamma=gamma, N=9)
        if self.worker_mean:
            cprintm("(+) worker mean")
            ds['mean_prob_choose'] = ds['mean_prob_choose'].mean('worker_id')
            ds = ds.drop_dims('worker_id')
        return ds
    
    def setup_behavior_data(self, ds1, ds):
        return ds1


class GenV6_Plotter_9Way(GenV5_Plotter_9Way):     
    pass


class GenV7_Plotter_9Way(GenV5_Plotter_9Way):     
    def setup_behavior_data(self, ds1, ds):
        # Treat case of mild renaming of class_name field
        if not np.array_equal(ds.class_name.values, ds1.choice.values):
            class_names1 = np.array([os.path.basename(str(class_name)) for class_name in ds.class_name.values])
            assert np.array_equal(class_names1, ds1.choice.values), (ds1.choice.values, class_names1)
            ds1 = ds1.assign_coords(choice=ds.class_name.values)
        return ds1


class GenV8_Plotter_9Way(GenV5_Plotter_9Way):     
    pass


########################################################
################## UNTARGETED ATTACKS ##################
########################################################


class Plotter_Untargeted_9Way(Plotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_inferred_catch = False
    
    def get_df_Human_Model_2AFC(self, pred_field=None, surrogate_str=''):
        ds = self.ds
        
        ds_list = [ds.sel(model_subject_name=ds.model_name + surrogate_str).drop_dims(['worker_id'], errors='ignore')]
        # Behavioral data might have worker_id. Avoid replicating other data_vars 
        if 'worker_id' in ds.coords:
            ds_list.append(ds.drop_dims(['model_subject_name']))
        df = pd.concat([x.to_dataframe().reset_index() for x in ds_list]).reset_index(drop=True)
        
        return df
    
    def compute_derivarives(self, ds):
        # Change class_index => class_name
        ds = ds.assign_coords(class_index=[self.class_dict[i] for i in ds.class_index.values]).rename(class_index='class_name')
        
        pred_logit_mean = ds.pred_logit.mean('class_name')
        ds['pred_class'] = xr.apply_ufunc(lambda x: self.class_dict[x], 
                                          ds.pred_logit.fillna(0).argmax('class_name'), 
                                          vectorize=True).where(~pred_logit_mean.isnull())
        ds['is_incorrect'] = ds.orig_class_name != ds.pred_class
        return ds
    
    def get_df_Model_Human_Alignment(self, 
                                     pred_field='pred_prob_choose_target_v2_ts0.01_cal',
                                     attacker_type_only=True):
        ds = self.ds
        ds1 = ds.copy()
        if attacker_type_only:
            # Include as subject models only models from attacker type (including attacker)
            ds1[pred_field] = ds1[pred_field].where(((ds1.model_subject_name == ds1.model_name) | 
                                                    (ds1.model_subject_name == ds1.model_name + '_v2') | 
                                                    (ds1.model_name == 'contrast_blend')))
        ds_list = [ds1.drop_dims(['model_subject_name', 'worker_id'], errors='ignore').merge(ds1[pred_field].to_dataset('model_subject_name'))]
        # Behavioral data might have worker_id. Avoid replicating other data_vars 
        if 'worker_id' in ds1.coords:
            ds_list.append(ds1.drop_dims(['model_subject_name']))
        df = pd.concat([x.to_dataframe().reset_index() for x in ds_list]).reset_index(drop=True)
        return df
    
    def plot_Alignment(self,
                       pred_field='is_incorrect',
                       attacker_type_only=False,
                       ylabel="Error Rate",
                       bbox_to_anchor=(1.5, .8),
                       custom_palette=None,
                       **kwargs):
        df = self.get_df_Model_Human_Alignment(pred_field, attacker_type_only)
        obj = AlignmentPlot(df,
                            x=self.x_var,
                            model_var=self.make_model_var(self.ds),
                            add_inferred_catch=self.add_inferred_catch,
                            )
        return obj.plot(concise=False, ylabel=self.adjust_ylabel(ylabel), bbox_to_anchor=bbox_to_anchor,
                        custom_palette=custom_palette, **kwargs)

    def plot_Model_Human_Alignment(self,
                                   pred_field='is_incorrect',
                                   attacker_type_only=True,
                                   ylabel="Error Rate",
                                   bbox_to_anchor=(1.5, .8),
                                   style=None,
                                   custom_palette=None, 
                                   **kwargs):
        df = self.get_df_Model_Human_Alignment(pred_field, attacker_type_only)
        obj = Model_Human_Alignment(df,
                                    x=self.x_var,
                                    model_var=self.make_model_var(self.ds),
                                    human_field='mean_error_rate',
                                    add_inferred_catch=self.add_inferred_catch,
                                    )
        return obj.plot(concise=False, ylabel=self.adjust_ylabel(ylabel), bbox_to_anchor=bbox_to_anchor,
                        style=style, custom_palette=custom_palette, **kwargs)
        

class GenV4_Plotter_Untargeted_9Way(Plotter_Untargeted_9Way):
    def add_behavior_data(self, ds, behavior_data_name):
        ds1 = xr.open_dataset(f"{PROJECT_ROOT}/results/behavior/{behavior_data_name}")
        if 'worker_id' not in ds1.k.dims:
            strs = behavior_data_name.split('/')
            behavior_data_name1 = '/'.join(strs[:-1] + [strs[-1].replace('flat', 'flat_collapsed_workers')])
            behavior_data_path1 = f"{PROJECT_ROOT}/results/behavior/{behavior_data_name1}"
            force_override = False
            if os.path.exists(behavior_data_path1) and not force_override:
                cprintm(f"Loading collapsed file: {behavior_data_name1}")
                ds1 = xr.open_dataset(behavior_data_path1)
            else:
                ds1 = self.get_collapsed_worker_id(behavior_data_name)
                cprintm(f"Saving {behavior_data_name1}")
                ds1.to_netcdf(behavior_data_path1)

        ds['n_selected'] = xr.DataArray(ds1.k.sel(stimulus_id=ds.image_id, choice=ds.class_name).data, 
                                        coords=dict(
                                            worker_id=ds1.worker_id,
                                            image_id=ds.image_id, 
                                            class_name=ds.class_name,
                                        ))
        ds['n_reps'] = xr.DataArray(ds1.n.sel(stimulus_id=ds.image_id).data,
                                    coords=dict(
                                        worker_id=ds1.worker_id,
                                        image_id=ds.image_id, 
                                    ))
        ds['mean_prob_choose'] = ds.n_selected / ds.n_reps
        ds['mean_prob_choose_target'] = ds['mean_prob_choose'].sel(class_name=ds.orig_class_name)
        if self.lapse_rate_norm:
            ds_calib = xr.open_dataset(f"{PROJECT_ROOT}/results/behavior/{behavior_data_name.replace('flat', 'calibration')}")
            catch_perfs = ds_calib.kcatch / ds_calib.ncatch
            gamma = LapseRateNorm.compute_gamma(catch_perfs, N=9)
            ds['mean_prob_choose_target'] = LapseRateNorm.apply_gamma(ds['mean_prob_choose_target'], gamma=gamma, N=9)
        ds['mean_error_rate'] = 1 - ds['mean_prob_choose_target']
        if self.worker_mean:
            cprintm("(+) worker mean")
            ds['mean_error_rate'] = ds['mean_error_rate'].mean('worker_id')
            ds = ds.drop_dims('worker_id')
        return ds
    