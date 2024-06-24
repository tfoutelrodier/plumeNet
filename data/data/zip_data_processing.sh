#!/bin/bash

# This scripts generates a basic report of the dataset as a csv
# columns are:
#     - species_name_snake_case
#     - original_species_name
#     - dataset_part
#     - folder_path_in_zip_file
#     - nb_images


# usage: zip_data_processing.sh zip_file.zip
zip_file=$1

good_file_extension="jpg"
dataset_part=$(basename "${zip_file}" .zip)
output_file=${dataset_part}_image_data.csv

echo "bird_name,bird_name_raw,dataset_part,folder,nb_images" > "${output_file}"

# Process zip file
current_folder=""
current_folder_image_count=0
current_bird_name=""
current_bird_name_snake_case=""
zipinfo -1 "${zip_file}" | while read -r line; do
    folder=$(dirname "${line}")
    file_name=$(basename "${line}")
    extension="${file_name##*.}"
    
    # Skip files with wrong extension
    if [ "${extension}" != "${good_file_extension}" ]; then
        continue
    fi

    # Check if folder changed
    if [ "${folder}" = "${current_folder}" ]; then
        ((current_folder_image_count++))
    else
        if [ "${current_folder}"  != "" ]; then
            echo "$current_folder ::: $current_bird_name_snake_case ::: $current_folder_image_count"
            echo "${current_bird_name_snake_case},${current_bird_name},${dataset_part},${current_folder},${current_folder_image_count}" >> "${output_file}"
        fi
        current_folder_image_count=1
        current_folder=${folder}

        # Parse bird name
        current_bird_name=$(basename "${folder}" | sed 's/^0\. //' | sed 's/^ //')
        current_bird_name_snake_case=$(echo "$current_bird_name" | tr '[:upper:]' '[:lower:]' | sed 's/[ -]/_/g' | sed "s/'//g")            
    fi
    
done
# last printing
echo "${current_bird_name_snake_case},${current_bird_name},${dataset_part},${current_folder},${current_folder_image_count}" >> "${output_file}"

