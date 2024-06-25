#!/bin/bash

# Merge csv file from individual dataset parts into a single dataset
#
output_file=all_DIB_image_data.csv
head -n1 DIB-10K_1_image_data.csv > ${output_file}
awk 'FNR>1' DIB-10K_*_image_data.csv >> ${output_file}

