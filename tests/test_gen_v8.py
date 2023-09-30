from robustness import *
from robustness.modulate_triplets import *


exp_root = f"{PROJECT_ROOT}/results/cache/gen_v8"
n_choice = 2
n_class = 9
ds0 = xr.open_dataset((f"{exp_root}/meta.nc"))


def test_all_images_in_meta():
    num_images_disk = len(os.listdir(f"{exp_root}/images/"))    
    assert num_images_disk == len(ds0.image_id)


def test_all_class_used():
    assert len(unique(ds0.orig_class_name)) ==  n_class
    

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
    

def test_not_all_subject_models_equal_pred():
    """ Test for crazy lambda dictionary bug (mapped all models to the last one)
    """
    for field in ['pred_logit']:
        assert not (ds0[field].data.std(-1) == 0).all(), field
        assert not (ds0[field].sel(model_subject_name=[name for name in ds0.model_subject_name.values if name.endswith('0_v2')]).data.std(-1) == 0).all(), field
        