
#!/bin/bash

french_bird_data_file="french_birds_metadata.csv"


# extract single image



tar -xvf ${dataset_part}.tgz ${bird_folder}/*.jpg --strip-components 1
