#!/bin/bash

# Prepare a VM to train a CNN dataset on bird data
# load dataset, install required packages, create required folders

# main information
DATASET_NAME="french_bird_db_25_50"

BASE_DIR=$(pwd)  # should be home. It will beak the code if you change it

# Datasets import
DATASET_BUCKET="french_bird_db_25-50"
DATASET_MOUNT_DIR="/mnt/dataset"
DATASET_DIR=${BASE_DIR}/"dataset"


# Output model saving
MODEL_BUCKET="plumenet_models_weights"
MODEL_MOUNT_DIR="/mnt/model"
MODEL_DIR=${BASE_DIR}/"model"

SCRIPT_DIR=${BASE_DIR}/"plumeNet/src/model_training_vm"
VENV_DIR=${BASE_DIR}/.venv


python_version=3.10.12  # python version to install and setup as default


# Install python and venv
sudo apt-get update  # may be required

# install specific version of python
# build specific python version from source
apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git


wget https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz
tar -xf Python-${python_version}.tgz

cd Python-${python_version}
./configure --enable-optimizations
make -j 4  # 4 is CPU number here

make altinstall

# install pip 
sudo apt install python3-pip


# setup venv from built python version
cd ${BASE_DIR}
${BASE_DIR}/Python-${python_version}/python/python -m venv ${VENV_DIR}
source .venv/bin/activate 


# create folders
sudo mkdir ${DATASET_MOUNT_DIR}
sudo mkdir ${MODEL_MOUNT_DIR} 

mkdir ${DATASET_DIR} ${MODEL_DIR}

# Get access to dataset and model buckets
gsutil rsync -r gs://${DATASET_BUCKET} ${DATASET_MOUNT_DIR}
gsutil rsync -r gs://${MODEL_BUCKET} ${MODEL_MOUNT_DIR}

# untar dataset
compressed_dataset_file=${DATASET_MOUNT_DIR}/${DATASET_NAME}.tgz
tar -xzf ${compressed_dataset_file} --directory ${DATASET_DIR}


# install required packages and softwares
git clone https://github.com/tfoutelrodier/plumeNet 
# go to target branch with required scripts
cd plumeNet
git checkout feat/model_training_vm

pip install -r ${SCRIPT_DIR}/requirements.txt


# run model on dataset
chmod 770 ${SCRIPT_DIR}/train_dataset_25_50.py
python ${SCRIPT_DIR}/train_dataset_25_50.py ${DATASET_DIR}/${DATASET_NAME} 10 50 ${MODEL_DIR}/${DATASET_NAME}


# save model
gsutil rsync -r ${MODEL_DIR} gs://${MODEL_BUCKET}
