"""
    Combine modulate_triplets job array results to meta.nc and images.tar.gz files and place in shared directory
"""


from wormholes.perturb import *


def save_merged_results(output_folder):
    ds = xr.concat([xr.open_dataset(p) for p in natsorted(glob.glob(f"{output_folder}/meta_job*.nc"))], dim='image_id')
    silentremove(f"{output_folder}/meta.nc")
    ds.to_netcdf(f"{output_folder}/meta.nc")
    

@dataclass
class Config:
    gen_version: int
    
    
if __name__ == '__main__':
    cfg = pyrallis.parse(config_class=Config)
    output_folder = f"{PROJECT_ROOT}/results/cache/gen_v{cfg.gen_version}"
    cprint1(f"Output folder [{output_folder}]")
    
    cprint1("Save merged results...")
    save_merged_results(output_folder)
    
    cprintm('Done.')
