
from wormholes.perturb import *
from dataclasses import dataclass
import pyrallis


@dataclass
class GenConfig:
    # job array id
    num: int
    seed: int = 0
    batch_size: int = 50
    gen_version: int = 1


@pyrallis.wrap()
def main(cfg: GenConfig):
    cprintm(f'Job [{cfg.num}]')
    globals()[f'GenV{cfg.gen_version}'](cfg).run()
    cprintm('Done.')


if __name__ == '__main__':
    main()
    