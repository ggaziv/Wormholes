#!/bin/bash


for ver in v26 v25 v23 v21 v18 v17 v7
do 
    echo $ver
    tar czvf gen_${ver}_images.tar.gz -C /braintree/home/guyga/projects/robustness/results/cache/triplets_data_gen_$ver images
done
