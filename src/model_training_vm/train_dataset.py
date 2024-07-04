import logging
import os
import sys

import json
import numpy as np
import tensorflow as tf
import keras
from keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# directiory with all images, one folder per class
dataset_dir = sys.argv[1] 
nb_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
output_model_dir = sys.argv[4]
logs_dir = sys.argv[5]  # main dir where logs should be stored

dataset_name = os.path.basename(dataset_dir)
logs_folder_name = f'{dataset_name}_E{str(nb_epochs)}_BS{str(batch_size)}'
model_save_file_basename = os.path.join(output_model_dir, f'{dataset_name}_BS{str(batch_size)}.keras')
patience = 10
random_state = 0

# set logging
os.mkdir(os.path.join(logs_dir, logs_folder_name))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(logs_dir, logs_folder_name, "script_log.log"), encoding='utf-8', level=logging.DEBUG)
logger.info(f"Passed arguments: {sys.argv}")


def create_model(num_classes: int) -> 'Model':
    # Charger le modèle ResNet50 pré-entraîné avec les poids ImageNet
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

    # Ajouter une couche de pooling global pour réduire la dimensionalité
    x = base_model.output
    x_2 = GlobalAveragePooling2D()(x)

    x_3 = Dense(num_classes, activation='relu')(x_2)

    # Ajouter une couche dense pour effectuer la classification
    output = Dense(num_classes, activation='softmax')(x_3)

    # Créer le modèle final en combinant le modèle de base et les couches supplémentaires
    model = Model(inputs=base_model.input, outputs=output)

    # Geler les couches du modèle de base pour éviter de les entraîner
    for layer in base_model.layers:
        layer.trainable = False

    return model


logger.info('Loading datasets train and rest...')
ds_train = image_dataset_from_directory(dataset_dir, labels='inferred', image_size=(300, 300), validation_split=0.2, subset="training", seed=random_state, batch_size=batch_size)
ds_test = image_dataset_from_directory(dataset_dir, labels='inferred', image_size=(300, 300), validation_split=0.2, subset="validation", seed=random_state, batch_size=batch_size)
logger.info("Done !")

# Obtenir le nombre de classes à partir de ds
class_names = ds_train.class_names
num_classes = len(ds_train.class_names)
with open(f'{output_model_dir}/{dataset_name}_class_names.json', 'w') as f:
    json.dump(class_names, f)

# Convertir les étiquettes cibles en vecteurs one-hot
logger.info('Preprocessing')
preprocess_input = tf.keras.applications.resnet50.preprocess_input

ds_train_preprocessed = ds_train.map(lambda x, y: (preprocess_input(x), y))
ds_test_preprocessed = ds_test.map(lambda x, y: (preprocess_input(x), y))

# Run the model
tf.config.list_physical_devices()

# Charger le modèle ResNet50 pré-entraîné avec les poids ImageNet
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# Ajouter une couche de pooling global pour réduire la dimensionalité
x = base_model.output
x_2 = GlobalAveragePooling2D()(x)

# x_3 = Dense(num_classes, activation='relu')(x_2)
x_3 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_2)
x_3 = Dropout(0.5)(x_3)

# Ajouter une couche dense pour effectuer la classification
output = Dense(num_classes, activation='softmax')(x_3)

# Créer le modèle final en combinant le modèle de base et les couches supplémentaires
model = Model(inputs=base_model.input, outputs=output)

# Geler les couches du modèle de base pour éviter de les entraîner
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             TensorBoard(log_dir=f'{logs_dir}/{logs_folder_name}', histogram_freq=0, write_graph=True, write_images=True)]


# Entraîner le modèle sur votre dataset avec le callback personnalisé
logger.info('training_model')
for epoch in range(nb_epochs):
    logger.info(f"starting epoch {str(epoch)}")
    model_save_file = model_save_file_basename + f"_epoch{str(epoch)}.keras"
    history = model.fit(ds_train_preprocessed, epochs=1, callbacks=callbacks, validation_data=ds_test_preprocessed)

    # save model
    logger.info(f'saving model info for loop {str(epoch)}: {model_save_file}')
    model.save(model_save_file)
