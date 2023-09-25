#!/bin/bash


# Create an associative array to map the old version numbers to the new ones
declare -A version_map
version_map["v7"]="v2"
version_map["v17"]="v4"
version_map["v18"]="v5"
version_map["v21"]="v6"
version_map["v23"]="v7"
version_map["v25"]="v8"
version_map["v26"]="v9"


for ver in v7 v17 v18 v21 v23 v25 v26
do 
    echo $ver
    tar czvf gen_${version_map[$ver]}_images.tar.gz -C /braintree/home/guyga/projects/robustness/results/cache/triplets_data_gen_$ver images
done
