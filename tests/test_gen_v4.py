from robustness import *
from robustness.modulate_triplets import *


exp_root = f"{PROJECT_ROOT}/results/cache/gen_v4"
n_choice = 2
n_class = 9
ds0 = xr.open_dataset((f"{exp_root}/meta.nc"))


def test_no_missing_meta_files():
    file_paths = natsorted(glob.glob(f"{exp_root}/meta_job*.nc"))
    assert len(file_paths) - 1 == extract_from_string(basename(file_paths[-1]), 'job'), "Some meta_job*.nc files are missing"
    
    
def test_all_images_in_meta():
    num_images_disk = len(os.listdir(f"{exp_root}/images/"))    
    assert num_images_disk == len(ds0.image_id)
    

def test_all_class_used():
    assert len(unique(ds0.orig_class_name)) == len(unique(ds0.target_class_name)) == n_class
    

def test_orig_image_not_nan():
    assert (ds0.orig_name != 'nan').all()
    
    
def test_budget_usage_not_nan():
    assert (ds0.budget_usage != np.nan).all()
    
    
def test_class_balance_in_meta():
    assert len(unique(unique(ds0.orig_class_name, return_counts=True)[-1])) == 1
    assert len(unique(unique(ds0.target_class_name, return_counts=True)[-1])) == 1
        
        
def test_no_chopped_paths():
    assert all([len(name.item().split('.')) == 2 for name in ds0.orig_name])
    assert all([(name == np.nan or len(name.item().split('.'))) for name in ds0.target_image_path])
    