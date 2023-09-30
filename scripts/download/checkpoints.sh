#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$


RELEASE_PATH=https://gazivlee-wormholes.s3.amazonaws.com/checkpoints


if [ -d checkpoints/ ]; then
    rm -rf checkpoints/
fi

mkdir checkpoints
cd checkpoints

# Download checkpoints
wget $RELEASE_PATH/imagenet_l2_1_0.pt
wget $RELEASE_PATH/imagenet_l2_1_0_v2.pt

wget $RELEASE_PATH/imagenet_l2_3_0.pt
wget $RELEASE_PATH/imagenet_l2_3_0_v2.pt

wget $RELEASE_PATH/imagenet_l2_10_0.pt
wget $RELEASE_PATH/imagenet_l2_10_0_v2.pt

wget $RELEASE_PATH/imagenet_l2_20_0.pt
wget $RELEASE_PATH/imagenet_l2_20_0_v2.pt

wget $RELEASE_PATH/imagenet_l2_50_0.pt
wget $RELEASE_PATH/imagenet_l2_50_0_v2.pt

wget $RELEASE_PATH/imagenet_linf_4.pt
wget $RELEASE_PATH/imagenet_linf_8.pt

wget $RELEASE_PATH/imagenet_vanilla_v2.pt
wget $RELEASE_PATH/imagenet_l2_0_0.pt  # Sanity check adversarial training with zero budget (should be equivalent to vanilla)

cd ..
