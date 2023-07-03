from wormholes.tools.vis_tools import show_image_row
from wormholes.perturb import *


im_res = 224


def make_viewer(ds_path, *args, **kwargs):
    if 'gen_v2' in os.path.basename(os.path.dirname(ds_path)):
        return TripletViewerMetaClass(ds_path, *args, **kwargs)    
    else:
        return TripletViewerMeta(ds_path, *args, **kwargs)


class TripletViewer:
    def __init__(self, ds_path, target_color=(255, 0, 0), coord_name=None, Gen=None):
        if os.path.exists(ds_path):
            self.ds = xr.open_dataset(ds_path)
        else:
            self.ds = xr.concat([xr.open_dataset(p) for p in natsorted(glob.glob(f"{os.path.dirname(ds_path)}/meta_job*.nc"))], dim='image_id')
        if coord_name is None:    
            coords_names = list(self.ds.coords)
            if coords_names:
                self.coord_name = coords_names[0]
            else:
                self.coord_name = list(self.ds.dims)[0]
        else:
            self.coord_name = coord_name
        self.exp_root = os.path.dirname(ds_path)
        self.target_color = target_color
        if Gen is None:
            gen_version = extract_from_string(ds_path, 'gen_v')
            Gen = globals()[f'GenV{gen_version}']
        self.transform = Gen(data_root=os.path.expanduser("~/data")).transform
        
    def get_image(self, partial_path):
        partial_path = str(partial_path)
        if partial_path.startswith('ilsvrc'):
            img_path = f"{PROJECT_ROOT}/data/{partial_path}"
            return self.transform(np.array(Image.open(img_path)))
        elif (partial_path.startswith('OOD') or 
              partial_path.startswith('ANI')):
            img_path = f"{PROJECT_ROOT}/data/{partial_path}"
            return self.transform(np.array(Image.open(img_path).convert('RGB')))
        else:
            img_path = f"{self.exp_root}/images/{partial_path}"
            return np.array(Image.open(img_path)) 

    def screen_by_dict(self, dict_true, dict_false, n_select, seed):
        # self.print_dicts(dict_true, dict_false)
        ds = self.ds
        # Eliminate irrelevant coordinates
        ds = ds.sel(model_subject_name=ds.model_subject_name.values[0], class_index=0)
        for k, v in dict_true.items():
            ind = ds[k] == v
            if ind.any():
                ds = ds.where(ind, drop=True)
            else:
                ds = ds.where(xr.apply_ufunc(lambda s: s.startswith(v), ds[k].astype(str), vectorize=True), drop=True)
        for k, v in dict_false.items():
            ds = ds.where(ds[k] != v, drop=True)
        assert len(ds[self.coord_name]) >= n_select, "Found less data points than n_select"
        ds = ds.sel({self.coord_name: np.random.RandomState(seed).choice(ds[self.coord_name], n_select, replace=False)})
        return ds
        
    def print_dicts(self, dict_true, dict_false):
        cprint1('*' * 20)
        cprint1(f'dict_true: {dict_true}')
        cprint1(f'dict_false: {dict_false}')
        cprint1('*' * 20)
        
    def __call__(self, dict_true={}, dict_false={}, n_select=8, seed=0, title=None, filename=None):
        xlist, tlist, ylist = self.get_plot_meta(dict_true, dict_false, n_select, seed)
        show_image_row(xlist, ylist=ylist, 
                       tlist=tlist, 
                       fontsize=14,
                       title=title,
                       filename=filename)
        
            
class TripletViewerMeta(TripletViewer):
    def __init__(self, ds_path, **kwargs):
        super().__init__(ds_path, **kwargs)       
    
    def __call__(self, dict_true={}, dict_false={}, n_select=8, seed=0):
        xlist, tlist = self.get_plot_meta(dict_true, dict_false, n_select, seed)
        show_image_row(xlist, ylist=['Distract', 'Modulated', 'Target', 'Orig'], 
                       tlist=tlist, 
                       fontsize=14)

    def get_plot_meta(self, dict_true, dict_false, n_select, seed):
        ds = self.screen_by_dict(dict_true, dict_false, n_select, seed)
        
        xlist, tlist = ([] for _ in range(2))
        for coord in ds[self.coord_name]:
            ds1 = ds.sel({self.coord_name: coord})
            for k in ['distract_image_name', 'image_id', 'target_image_name', 'orig_name']:
                img = self.get_image(ds1[k].item())
                if k == 'image_id':
                    pad_size = 5
                    img[:pad_size] = img[-pad_size:] = img[:, -pad_size:] = img[:, :pad_size] = self.target_color
                xlist.append(img)
                if k == 'image_id':
                    tlist.append(r'($\delta=${:.1f}% | $\epsilon=${:.1f})'.format(ds1.budget_usage.item()*100, ds1.budget.item()))
                else:
                    tlist.append('')
            
        xlist = np.array(xlist).reshape(-1, 4, *xlist[0].shape).transpose(1, 0, 2, 3 ,4)
        tlist = list(np.array(tlist).reshape(-1, 4).T)
        return xlist, tlist
        

class TripletViewerMetaClass(TripletViewer):
    def __init__(self, ds_path, **kwargs):
        super().__init__(ds_path, **kwargs)       
    
    def __call__(self, dict_true={}, dict_false={}, n_select=8, seed=0):
        ds = self.screen_by_dict(dict_true, dict_false, n_select, seed)
        
        xlist, tlist = ([] for _ in range(2))
        for coord in ds[self.coord_name]:
            ds1 = ds.sel({self.coord_name: coord})
            for k in ['image_id', 'orig_name']:
                img = self.get_image(ds1[k].item())
                if k == 'image_id':
                    pad_size = 5
                    img[:pad_size] = img[-pad_size:] = img[:, -pad_size:] = img[:, :pad_size] = self.target_color
                xlist.append(img)
            title_strs = []
            for k in ['target_class_name', 'distract_class_name']:
                class_name = str(ds1[k].item()).split('_')[-1]
                if ds1.target_class_name.item().split('_')[-1] == class_name:
                    title_strs.append(r'$\bf{}$'.format(class_name))          
                else:
                    title_strs.append(class_name)
            tlist.append(' | '.join(title_strs) + '\n' + \
                r'($\delta=${:.1f}% | $\epsilon=${:.1f})'.format(ds1.budget_usage.item()*100, ds1.budget.item()))
            
        xlist = np.array(xlist).reshape(-1, 2, *xlist[0].shape).transpose(1, 0, 2, 3 ,4)
        
        show_image_row(xlist, ylist=['Modulated', 'Orig'], 
                       tlist=[tlist, [''] * len(tlist)], 
                       fontsize=14)
        
class TripletViewerMetaTokens(TripletViewer):
    def __init__(self, ds_path, **kwargs):
        self.custom_pattern = kwargs.pop('custom_pattern', r'$\rightarrow\bf{}$')
        super().__init__(ds_path, **kwargs, coord_name='image_id')
        
    def get_plot_meta(self, dict_true={}, dict_false={}, n_select=8, seed=0, pad_size=5):
        ds = self.screen_by_dict(dict_true, dict_false, n_select, seed)
        
        xlist, tlist = ([] for _ in range(2))
        for coord in ds[self.coord_name]:
            xlist_triplet = []
            ds1 = ds.sel({self.coord_name: coord})
            for k in [self.coord_name, 'orig_name']:
                img_path = ds1[k].item()
                if img_path not in ['nan', ''] or k == 'orig_name':
                    if img_path in ['nan', '']:  # Make random pixels
                        img = (np.random.rand(im_res, im_res, 3) * 255).astype('uint8')
                    else:
                        img = self.get_image(img_path)
                    if k == self.coord_name:
                        if pad_size > 0:
                            img[:pad_size] = img[-pad_size:] = img[:, -pad_size:] = img[:, :pad_size] = self.target_color
                    xlist_triplet.append(img)
            title_strs = [self.custom_pattern.format(os.path.basename(str(ds1.target_class_name.item()).replace('_', '-')))]
            tlist_triplet = [' | '.join(title_strs) + '\n' + \
                r'($\delta=${:.1f}% | $\epsilon=${:.1f})'.format(ds1.budget_usage.item()*100, ds1.budget.item())]
            orig_class_name = str(ds1.orig_class_name.item())
            tlist_triplet.append(orig_class_name if orig_class_name not in ['noise', 'UNI'] else 'noise (illust.)')
            tlist.extend(tlist_triplet)
            xlist.extend(xlist_triplet)
        
        xlist = np.array(xlist).reshape(-1, len(xlist_triplet), *xlist[0].shape).transpose(1, 0, 2, 3 ,4)
        tlist = list(np.array(tlist).reshape(-1, len(xlist_triplet)).T)
        ylist = ['Modulated', 'Orig']
        return xlist, tlist, ylist
    
    
class TripletViewerMetaTokensCompare(TripletViewerMetaTokens):
    def __call__(self, kwargs_list, n_cut=1, row_names=None, title=None, filename=None, fontsize=15):
        xlist, tlist, ylist = self.get_plot_meta(**kwargs_list[0])
        for kwargs in kwargs_list[1:]:
            xlist_cur, tlist_cur, ylist_cur = self.get_plot_meta(**kwargs)
            xlist = np.vstack([xlist_cur[:n_cut], xlist])
            tlist = tlist_cur[:n_cut] + tlist
            ylist = ylist_cur[:n_cut] + ylist
        if row_names is not None:
            assert len(ylist) == len(row_names)
            ylist = row_names
        show_image_row(xlist, 
                       ylist=ylist, 
                       tlist=tlist, 
                       fontsize=fontsize,
                       title=title,
                       filename=filename)
