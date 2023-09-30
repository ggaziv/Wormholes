"""
    General purpose utilities
"""


import os, types, itertools, uuid, random, re, glob, shutil
import numpy as np
import xarray as xr
from collections import namedtuple
from functools import partial
from multiprocessing.dummy import Pool
from termcolor import cprint
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
from argparse import Namespace
import pyrallis
from dataclasses import dataclass
identity = lambda x: x
cprint1 = lambda s, *args, **kwargs: cprint(s, 'cyan', attrs=['bold'], *args, **kwargs)
cprintm = lambda s, *args, **kwargs: cprint(s, 'magenta', *args, **kwargs)


def invert_dict(d):
    return {v:[k for k in d if d[k] == v] for v in d.values()}


def silentremove(file_or_folder_name):
    if os.path.exists(file_or_folder_name):
        if os.path.isfile(file_or_folder_name):
            os.remove(file_or_folder_name)
        else:  # Folder
            shutil.rmtree(file_or_folder_name)


def overridefolder(folder_path):
    silentremove(folder_path)
    os.makedirs(folder_path)
    return folder_path