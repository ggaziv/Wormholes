#!/bin/bash    


#SBATCH --mail-type=FAIL
#SBATCH --job-name=vanilla_v2
#SBATCH -o /home/guyga/slurm_reports/vanilla_v2_%N.%A.out
#SBATCH -N 1 -c 30
#SBATCH --gres=gpu:4 --constraint=16GB


DATA_PATH=<DATA_PATH>
RESULTS_DIR=<RESULTS_DIR>

DATASET=imagenet

python -m wormholes.main --dataset $DATASET --data $DATA_PATH \
   --adv-train 0 --arch resnet50 \
   --out-dir $RESULTS_DIR --exp-name imagenet_vanilla_v2