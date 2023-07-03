#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$


RELEASE_PATH=https://gazivlee-wormholes.s3.amazonaws.com


mkdir -p results/behavior
cd results/behavior


rm -f behavior_data.tar.gz*
wget $RELEASE_PATH/behavior_data.tar.gz
tar xvzf behavior_data.tar.gz .
rm -f behavior_data.tar.gz


cd ../..
