#!/bin/bash    


#SBATCH --mail-type=FAIL
#SBATCH --job-name=eps0
#SBATCH -o /home/guyga/slurm_reports/eps0_%N.%A.out
#SBATCH -N 1 -c 30
#SBATCH --gres=gpu:4 --constraint=16GB


DATA_PATH=<DATA_PATH>
RESULTS_DIR=<RESULTS_DIR>

DATASET=imagenet

python -m wormholes.main --dataset $DATASET --data $DATA_PATH \
   --adv-train 1 --arch resnet50 \
   --out-dir $RESULTS_DIR --exp-name imagenet_l2_0_0 --eps 0.0 --attack-lr 0.3 \
   --attack-steps 0 --constraint 2 #\
   # --resume-optimizer 1 --resume ${RESULTS_DIR}/imagenet_l2_0_0/checkpoint.pt.latest