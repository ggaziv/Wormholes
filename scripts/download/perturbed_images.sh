#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$


RELEASE_PATH=https://gazivlee-wormholes.s3.amazonaws.com


mkdir -p results/cache
cd results/cache


for ver in v2 v4 v5 v6 v7 v8 v9
do
    echo $ver
    mkdir -p gen_${ver}
    cd gen_${ver}
    rm -f gen_${ver}_images.tar.gz*
    wget $RELEASE_PATH/perturbed_images/gen_${ver}_images.tar.gz
    if [ -d images/ ]; then
        rm -rf images/
    fi
    tar xzf gen_${ver}_images.tar.gz
    rm -f gen_${ver}_images.tar.gz
    cd ..
done

cd ..
