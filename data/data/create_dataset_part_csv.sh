#!/bin/bash

csv_folder="."
data_folder="../raw_data/IDB"
merged_csv_file=all_DIB_image_data.csv
cd csv_folder

for i in $(seq 1 70); do
    echo "$i"
    data_file=${data_folder}/DIB-10K_${i}.tgz  # input
    csv_file=DIB-10K_${i}_image_data.csv  # output
    
    if [ ! -f  "${csv_file}" ]; then
        echo "csv file don't exist"
        ./DIB_tgz_processing.sh ${data_file}
    else
        echo "csv file already exist"
    fi

    # merge in big csv
    if [[ "${i}" -eq 1 ]]; then
        head -n 1 "${csv_file}" > ${merged_csv_file}
    fi
    awk 'NR > 1' ${csv_file} >> ${merged_csv_file}
done