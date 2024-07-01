import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from keras.utils import image_dataset_from_directory
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

# directiory with all images, one folder per class
dataset_name = sys.argv[0] 
nb_epochs = sys.argv[1]
batch_size = sys.argv[2]
output_model_dir = sys.argv[3]

dataset_dir = dataset_name
logs_folder_name = f'{dataset_name}_{nb_epochs}_{batch_size}'
model_save_file = os.path.join(output_model_dir, f'{dataset_name}_{nb_epochs}_{batch_size}.keras')
patience = 10
random_state = 0


def create_model() -> 'Model':
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


print('Loading datasets traina and rest...')
ds_train = image_dataset_from_directory(dataset_dir, labels='inferred', image_size=(300, 300), validation_split=0.2, subset="training", seed=random_state)
ds_test = image_dataset_from_directory(dataset_dir, labels='inferred', image_size=(300, 300), validation_split=0.2, subset="validation", seed=random_state)
print("Done !")

# Obtenir le nombre de classes à partir de ds
num_classes = len(ds_train.class_names)

# Convertir les étiquettes cibles en vecteurs one-hot
print('One-hot-encoding')
ds_train = ds_train.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
ds_test = ds_test.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

# Run the model
tf.config.list_physical_devices()

model = create_model()

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             TensorBoard(log_dir=f'./logs/{logs_folder_name}', histogram_freq=0, write_graph=True, write_images=True)]


# Entraîner le modèle sur votre dataset avec le callback personnalisé
history = model.fit(ds_train, epochs=nb_epochs, callbacks=callbacks, validation_data=ds_test)

# save model to pickle
model.save(model_save_file)

