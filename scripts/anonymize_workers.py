#!/usr/bin/env python3


import xarray as xr 
import numpy as np
import os, uuid
import sys


ROOT_PATH = "/braintree/home/guyga/projects/Wormholes/results/behavior"

    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        versions = sys.argv[1:]
    else:   
        versions = ['v25', 'v23', 'v21', 'v18', 'v17']
        
    for ver in versions:
        print(f"Anonymizing {ver}...")
        file_path = f"{ROOT_PATH}/gen_{ver}/ds_calibration.nc"
        with xr.open_dataset(file_path) as ds:
            worker_id_dict = {worker_id: uuid.uuid4().hex for worker_id in np.unique(ds.worker_id)}
            ds['worker_id'] = xr.apply_ufunc(lambda x: worker_id_dict[x], ds['worker_id'], vectorize=True)
            ds.to_netcdf(file_path.replace('.nc', '_anon.nc'))
        # Swap files
        os.remove(file_path)
        os.rename(file_path.replace('.nc', '_anon.nc'), file_path)
        
        file_path = f"{ROOT_PATH}/gen_{ver}/ds_flat.nc"
        with xr.open_dataset(file_path) as ds:
            ds['worker_id'] = xr.apply_ufunc(lambda x: worker_id_dict[x], ds['worker_id'], vectorize=True)
            ds.to_netcdf(file_path.replace('.nc', '_anon.nc'))
        # Swap files
        os.remove(file_path)
        os.rename(file_path.replace('.nc', '_anon.nc'), file_path)
    print("Done.")