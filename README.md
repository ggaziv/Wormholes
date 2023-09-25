# Robustified ANNs Reveal Wormholes <br>Between Human Category Percepts (NeurIPS 2023)


<div align="left">
  <img width="90%" src=".github/interp_cycle.gif">
</div>

### ***Official implementation & data for:***
> **Robustified ANNs Reveal Wormholes Between Human Category Percepts (NeurIPS 2023)** \
> Guy Gaziv*, Michael J. Lee*, James J. DiCarlo  
> 
> **TL;DR**  
> Robustified ANNs reliably discover image perturbations in the "human-presumed-stable" pixel-budget regime that can strongly disrupt human behavior, or can precisely drive human perceptual state transitions to prescribed target states.

<a href="https://doi.org/10.1016/j.neuroimage.2022.119121" target="_blank"><img src="https://img.shields.io/badge/NeurIPS-paper-b31b1b.svg" height=22.5></a>
<a href="https://himjl.github.io/pwormholes/" target="_blank"><img src="https://img.shields.io/badge/-Project Page-blue.svg" height=22.5></a>
<a href="https://video.tau.ac.il/events/index.php?option=com_k2&view=item&id=10112:fmri&Itemid=550" target="_blank"><img src="https://img.shields.io/badge/-Summary Video-purple.svg" height=22.5></a>


###
<div align="left">
  <img width="90%" src=".github/main_results.gif">
</div>


## Table of Contents
- [Setup](#setup)
  * [Data & Results](#data--results)
  * [Environment](#environment)
  * [Test code](#test-code-runs-successfully)
- [Paper results](#viewing-paper-results)
- [Generating images](#perturbing-images)
- [Training models](#training-models)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Setup
### Environment
Clone this repo and create a designated conda environment using the `env.yml` file:
```
cd Wormholes
conda env create -n wormholes -f env.yml
conda activate wormholes
```
*Recommended: for quick environment solving use [libmamba-solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) or mamba directly:*
```
conda install -y -c conda-forge mamba
```
(Then just use `mamba` instead of `conda` in the above calls)

###  Data & Results

We make all our data and results publicly available. 
This includes source data, model checkpoints, perturbed images with their metadata, and the MTurk behavioral reports (anonymized).
For convenience, we include download scripts for each component under `./scripts/download`.
**The following convenience script sets up all data automatically.**
```
./scripts/download/all.sh
```

Finally, download [ILSVRC](https://image-net.org/challenges/LSVRC/index.php) and place a link to it under `data/`:
```
ln -s <ILSVRC-FOLDER> data/
```

Notably, for completeness, some of the checkpoints are direct copies of those published in [robustness](https://github.com/MadryLab/robustness#pretrained-models).
'v2' versions of models are equivalent to the non-'v2' versions but initialized and trained using a different seed.

### Test code runs successfully
```
export CUDA_VISIBLE_DEVICES=<CUDA_ID>
pytest -sv tests/test_gen_smoketest.py
```


## Viewing paper results
We include two notebooks for visualization for results under `notebooks/`:
- `images_view.ipynb` -- visualizing image perturbation results with their metadata. Note that this notebook assumes having [ILSVRC](https://image-net.org/challenges/LSVRC/index.php) validation set, as `.data/ilsvrc/val`.
- `results_figures.ipynb` -- replicating all paper figures.

To work with these notebooks, you might need to install the jupyter kernel of the environment:
```
python -m ipykernel install --user --name "wormholes" --display-name wormholes
```


## Perturbing images
All our paper related image perutbation experiments are implemented under `wormholes/perturb/`.
To run an experiment `gen_v<X>.py`:
```
export CUDA_VISIBLE_DEVICES=<CUDA_ID>
./scripts/gen_data.sh <X>
```
Results will output under `results/cache/` (note they might already exist there if you downloaded them).
Lastly you will need to merge the generated metadata to a single file:
```
python scripts/combine_results.py --gen_version <X>
```


## Training models
Scripts for training models and surrogate models, similar to those in the paper (albeit from a different seed), are included under `scripts/train/`.


## Acknowledgments
This code borrows from [robustness](https://github.com/MadryLab/robustness).


## Citation
If you find this repository useful, please consider giving a star ⭐️ and citation:
```
@article{Gaziv2023,
  title = {Robustified ANNs Reveal Wormholes Between Human Category Percepts},
  author = {Guy Gaziv and Michael J. Lee and James J. DiCarlo},
  url = {https://arxiv.org/abs/2308.06887v1},
  year = {2023},
}
```
