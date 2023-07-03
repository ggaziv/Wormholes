from wormholes.perturb import *
import pytest
from tempfile import TemporaryDirectory, NamedTemporaryFile
from argparse import Namespace


@pytest.mark.parametrize('gen_version', [7, 17, 18, 21, 23, 25, 26])
@pytest.mark.parametrize('num', [0, 10, 13])
def test_gen(gen_version, num, batch_size=5):
    cfg = Namespace(num=num, seed=0, batch_size=batch_size, gen_version=gen_version)
    gen_obj = globals()[f'GenV{gen_version}'](cfg)
    gen_obj.interp_hparams_tup_list = []
    
    get_ds = gen_obj.get_ds
    def get_ds_mocked(self):
        ds = get_ds()
        ds['n_iter'] = 3
        return ds
    gen_obj.get_ds = types.MethodType(get_ds_mocked, gen_obj)
    
    screen_job_work = gen_obj.screen_job_work
    def screen_job_work_mocked(self, df):
        group, df_agg = screen_job_work(df, print=False)
        df_agg = df_agg[:batch_size]
        return group, df_agg
    gen_obj.screen_job_work = types.MethodType(screen_job_work_mocked, gen_obj)
    
    with TemporaryDirectory() as output_folder:
        gen_obj.output_folder = output_folder
        gen_obj.run()