#!/bin/bash

# Prepare a VM to train a CNN dataset on bird data
# load dataset, install required packages, create required folders



# main information
DATASET_NAME="french_bird_db_25_50"


# Datasets import
DATASET_BUCKET="french_bird_db_25_50"
LOCAL_DATASET_MOUNT_DIR="/mnt/bucket/dataset"
LOCAL_DATASET_DIR="/mnt/dataset"


# Output model saving
MODEL_BUCKET="plumenet_model_weights"
LOCAL_MODEL_MOUNT_DIR="/mnt/bucket/model"
LOCAL_MODEL_DIR="/mnt/model"


# Get access to dataset and model buckets  
gsutil rsync -r gs://${DATASET_BUCKET} ${LOCAL_DATASET_MOUNT_DIR}
gsutil rsync -r gs://${MODEL_BUCKET} ${LOCAL_MODEL_MOUNT_DIR}
gsutil rsync -r gs://${SCRIPT_BUCKET} ${LOCAL_MODEL_MOUNT_DIR}



# untar dataset
compressed_dataset_file=${LOCAL_DATASET_MOUNT_DIR}/${DATASET_NAME}.tgz
tar -xzf ${compressed_dataset_file} --directory ${LOCAL_DATASET_DIR}


# install required packages and softwares
gh clone repo tfoutelrodier/plumeNet


# run model on dataset


# save model
gsutil rsync -r ${LOCAL_MODEL_DIR} gs://${MODEL_BUCKET}
