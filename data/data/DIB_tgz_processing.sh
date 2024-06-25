#!/bin/bash

# for i in $(seq 1 10); do
#     file=DIB-10K_${i}.tgz
#     tar -ztf ${file} | head
# done 



# usage: DIB_tgz_processing.sh targz_file.tgz
targz_file=$1

good_file_extension="jpg"
dataset_part=$(basename "${targz_file}" .tgz)
output_file=${dataset_part}_image_data.csv

echo "bird_name,bird_name_raw,dataset_part,folder,nb_images" > "${output_file}"

# Process input file
current_folder=""
current_folder_image_count=0
current_bird_name=""
current_bird_name_snake_case=""
tar -ztf "${targz_file}" | while read -r line; do
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
        current_bird_name=$(basename "${folder}" | sed 's/^[0-9]*\.//' | sed 's/^ +//' | sed 's/ +$//')
        current_bird_name_snake_case=$(echo "$current_bird_name" | tr '[:upper:]' '[:lower:]' | sed 's/[ -]/_/g' | sed "s/'//g")            
    fi
    
done
# last printing
echo "${current_bird_name_snake_case},${current_bird_name},${dataset_part},${current_folder},${current_folder_image_count}" >> "${output_file}"
