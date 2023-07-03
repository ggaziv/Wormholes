#!/bin/bash

source_folder="/braintree/home/guyga/data/shared"
tmp_folder="/tmp/guyga_tmp_folder"
tar_file="perturbed_data.tar.gz"

included_dirs=("v26" "v25" "v23" "v21" "v18" "v17" "v7")

mkdir -p $tmp_folder
for dir in "${included_dirs[@]}"; do
  mkdir -p "$tmp_folder/gen_$dir"
  rsync -arogv $source_folder/triplets_data_gen_$dir/meta.nc "$tmp_folder/gen_$dir/"
done

tar -czf "$tar_file" -C $tmp_folder .
rm -rf $tmp_folder


# Useful code snippets:
# ssh braintree-gpu-3 'tar cfz --exclude="*images*" - -C /braintree/home/guyga/projects/Wormholes/results/cache .' > perturbed_data.tar.gz
