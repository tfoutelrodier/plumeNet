#!/bin/bash

# Merge csv file from individual dataset parts into a single dataset
#
output_file=all_bird_species_image_data.csv
head -n1 part_1_image_data.csv > ${output_file}
awk 'FNR>1' part_*_image_data.csv >> ${output_file}

