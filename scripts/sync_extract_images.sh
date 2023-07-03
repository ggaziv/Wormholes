#!/bin/bash


for ver in v26 v25 v23 v21 v18 v17 v7; do tar xzf /braintree/home/guyga/data/shared/triplets_data_gen_${ver}/images.tar.gz -C gen_${ver}/ --strip-components 10; done
# for ver in v26 v25 v23 v21 v18 v17 v7; do rsync -arogv /braintree/home/guyga/data/shared/triplets_data_gen_${ver}/meta.nc -C gen_${ver}/ ; done