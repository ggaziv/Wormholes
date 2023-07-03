#!/bin/bash


BEHAVIOR_PATH=/braintree/home/guyga/projects/robustness/results/behavior
OUTPUT_PATH=/braintree/home/guyga/projects/Wormholes/results/behavior


if [ -d $OUTPUT_PATH ]; then
    rm -rf $OUTPUT_PATH
fi


for ver in v25 v23 v21 v18 v17
do
    echo $ver
    rm -rf ${OUTPUT_PATH}/gen_${ver}
    mkdir -p ${OUTPUT_PATH}/gen_${ver}
    rsync -arog ${BEHAVIOR_PATH}/triplets_data_gen_${ver}/ds_calibration_triplets_data_gen_${ver}.nc ${OUTPUT_PATH}/gen_${ver}/ds_calibration.nc
    rsync -arog ${BEHAVIOR_PATH}/triplets_data_gen_${ver}/ds_flat_triplets_data_gen_${ver}.nc ${OUTPUT_PATH}/gen_${ver}/ds_flat.nc
    python -m scripts.anonymize_workers $ver
done


