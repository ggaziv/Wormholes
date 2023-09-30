#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$


RELEASE_PATH=https://gazivlee-wormholes.s3.amazonaws.com/source_data


mkdir -p data
cd data

# Download custom datasets
for dset_name in OOD ANI ANI2 AnimalImageDataset 
do  
    rm -f ${dset_name}.tar.gz*
    wget $RELEASE_PATH/${dset_name}.tar.gz
    if [ -d $dset_name ]; then
        rm -rf $dset_name
    fi
    mkdir $dset_name
    tar xvzf ${dset_name}.tar.gz -C $dset_name
    rm -f ${dset_name}.tar.gz
done

cd ..
