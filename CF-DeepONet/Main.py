# -*- coding: utf-8 -*-

"""
File: Main.py
Author: Jing-lai Zheng
Date: 2024-10-01
Description: This is the training code of CF-DeepONet for the Laval nozzle case.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
from tensorflow.keras.initializers import GlorotUniform
import pandas as pd

# Define seed and GPU
seed = 2024
tf.keras.utils.set_random_seed(seed)
gpus = tf.config.list_physical_devices('GPU')

# Define activation function
actf = 'relu'


# Define branch net（input NPR）
def build_branch_net(input_dim, output_dim):
    inputs = layers.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(inputs)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    outputs = layers.Dense(output_dim, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    return models.Model(inputs, outputs)


# Define trunk net（input x y）
def build_trunk_net(input_dim, output_dim):
    inputs = layers.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(inputs)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    x = layers.Dense(100, activation=actf, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    outputs = layers.Dense(output_dim, kernel_initializer=GlorotUniform(), dtype=tf.float32)(x)
    return models.Model(inputs, outputs)


# Define CF-DeepONet (a branch net and a trunk net)
def build_CF_DeepONet(branch_net, trunk_net):
    input_branch = layers.Input(shape=(branch_net.input_shape[1],), dtype=tf.float32)
    input_trunk = layers.Input(shape=(trunk_net.input_shape[1],), dtype=tf.float32)
    branch_out = branch_net(input_branch)
    trunk_out = trunk_net(input_trunk)
    outputs = tf.reduce_sum(branch_out * trunk_out, axis=1, keepdims=True)
    model = models.Model([input_branch, input_trunk], outputs)
    return model


# Load data function
def load_data(file_path, key):
    df = pd.read_hdf(file_path, key=key)
    npr = df['NPR'].values.reshape(-1, 1)  # Load NPR as branch net input
    xy = df[['X Coordinate', 'Y Coordinate']].values  # Load x y as trunk net input
    temperature = df['Temperature'].values.reshape(-1, 1)  # Temperature label
    return npr, xy, temperature


# Load training data
train_npr, train_xy, train_temperature = load_data('Laval_dataset_train.h5', key='train')

# Load testing data
test_npr, test_xy, test_temperature = load_data('Laval_dataset_test.h5', key='test')

# Normalization
scaler_npr = MinMaxScaler()
scaler_xy = MinMaxScaler()
scaler_temperature = MinMaxScaler()

train_npr = scaler_npr.fit_transform(train_npr)
train_xy = scaler_xy.fit_transform(train_xy)
train_temperature = scaler_temperature.fit_transform(train_temperature)
test_npr = scaler_npr.transform(test_npr)
test_xy = scaler_xy.transform(test_xy)
test_temperature = scaler_temperature.transform(test_temperature)

# Save the normalizer
joblib.dump(scaler_npr, 'scaler_npr.pkl')
joblib.dump(scaler_xy, 'scaler_xy.pkl')
joblib.dump(scaler_temperature, 'scaler_temperature.pkl')

# Print shape log
print(f"Train NPR shape: {train_npr.shape}")
print(f"Train (x,y) shape: {train_xy.shape}")
print(f"Train Temperature shape: {train_temperature.shape}")
print(f"Test NPR shape: {test_npr.shape}")
print(f"Test (x,y) shape: {test_xy.shape}")
print(f"Test Temperature shape: {test_temperature.shape}")

input_dim_branch = train_npr.shape[1]  # Shape of branch net input
input_dim_trunk = train_xy.shape[1]  # Shape of trunk net
output_dim = 100  # Shape of output feature vector

# Instantiate network
branch_net = build_branch_net(input_dim_branch, output_dim)
trunk_net = build_trunk_net(input_dim_trunk, output_dim)
CF_DeepONet = build_CF_DeepONet(branch_net, trunk_net)

# Print net summary
CF_DeepONet.summary()

# Define learning rate schedule
initial_learning_rate = 8e-4
decay_steps = 10000
decay_rate = 0.95

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Compile and train
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
CF_DeepONet.compile(optimizer=adam_optimizer, loss='mse')
start_time = time.time()
history = CF_DeepONet.fit([train_npr, train_xy], train_temperature, epochs=10000,
                          batch_size=int(len(train_npr)/5),
                          validation_data=([test_npr, test_xy], test_temperature))
end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time} seconds")

# Save model
CF_DeepONet.save('model_temperature.h5')

# Save loss
np.save('history_temperature.npy', history.history)