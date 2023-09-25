#!/bin/bash

source_folder="/braintree/home/guyga/data/shared"
tmp_folder="/tmp/guyga_tmp_folder"
tar_file="perturbed_data.tar.gz"


# Create an associative array to map the old version numbers to the new ones
declare -A version_map
version_map["v7"]="v2"
version_map["v17"]="v4"
version_map["v18"]="v5"
version_map["v21"]="v6"
version_map["v23"]="v7"
version_map["v25"]="v8"
version_map["v26"]="v9"

included_vers=("v26" "v25" "v23" "v21" "v18" "v17" "v7")

mkdir -p $tmp_folder
for ver in "${included_vers[@]}"; do
  mkdir -p "$tmp_folder/gen_${version_map[$ver]}"
  rsync -arogv $source_folder/triplets_data_gen_$ver/meta.nc "$tmp_folder/gen_${version_map[$ver]}/"
done

tar -czf "$tar_file" -C $tmp_folder .
rm -rf $tmp_folder


# Useful code snippets:
# ssh braintree-gpu-3 'tar cfz --exclude="*images*" - -C /braintree/home/guyga/projects/Wormholes/results/cache .' > perturbed_data.tar.gz
