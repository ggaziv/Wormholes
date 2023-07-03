#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$


RELEASE_PATH=https://gazivlee-wormholes.s3.amazonaws.com


mkdir -p results/cache
cd results/cache


rm -f perturbed_data.tar.gz*
wget $RELEASE_PATH/perturbed_data.tar.gz
tar xvzf perturbed_data.tar.gz .
rm -f perturbed_data.tar.gz


cd ../..
