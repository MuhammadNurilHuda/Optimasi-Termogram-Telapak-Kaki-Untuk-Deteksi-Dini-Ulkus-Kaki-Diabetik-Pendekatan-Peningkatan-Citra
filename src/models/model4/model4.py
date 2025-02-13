# src/models/model4/model4.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape_image, input_shape_tabular):
    """
    Membuat arsitektur model yang menggabungkan data citra (kiri dan kanan) dan data tabular dengan arsitektur yang berbeda dari model sebelumnya.

    Parameters
    ----------
    input_shape_image : tuple
        Ukuran input untuk citra, dalam format (height, width, channels).
    input_shape_tabular : int
        Ukuran input untuk data tabular (jumlah fitur tabular).
    
    Returns
    -------
    model : tf.keras.Model
        Model Keras yang telah dibangun.
    """
    # Input citra kiri
    input_left = layers.Input(shape=input_shape_image, name='input_left')

    conv1_left = layers.Conv2D(64, (3, 3), activation='relu')(input_left)
    pool1_left = layers.MaxPooling2D(pool_size=(2, 2))(conv1_left)
    batch1_left = layers.BatchNormalization()(pool1_left)
    
    flatten_left = layers.Flatten()(batch1_left)

    # Input citra kanan
    input_right = layers.Input(shape=input_shape_image, name='input_right')

    conv1_right = layers.Conv2D(128, (3, 3), activation='relu')(input_right)
    pool1_right = layers.MaxPooling2D(pool_size=(2, 2))(conv1_right)
    batch1_right = layers.BatchNormalization()(pool1_right)
    
    flatten_right = layers.Flatten()(batch1_right)

    # Input data tabular
    input_tabular = layers.Input(shape=(input_shape_tabular,), name='input_tabular')
    dense1_tabular = layers.Dense(64, activation='relu')(input_tabular)
    dense2_tabular = layers.Dense(128, activation='relu')(dense1_tabular)
    dense3_tabular = layers.Dense(256, activation='relu')(dense2_tabular)
    drop1_tabular = layers.Dropout(0.5)(dense3_tabular)

    # Menggabungkan fitur dari citra dan data tabular
    concat = layers.concatenate([flatten_left, flatten_right, drop1_tabular], name='concatenate')

    # Layer fully connected untuk output gabungan
    dense1 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2())(concat)
    drop1 = layers.Dropout(0.2)(dense1)

    dense2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2())(drop1)
    drop2 = layers.Dropout(0.2)(dense2)

    dense3 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2())(drop2)
    drop3 = layers.Dropout(0.2)(dense3)

    # Output akhir
    output = layers.Dense(1, activation='sigmoid', name='output')(drop3)

    # Membangun model dengan beberapa input
    model = models.Model(inputs=[input_left, input_right, input_tabular], outputs=output, name='Model4')
    return model