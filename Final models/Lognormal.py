import pandas as pd
import numpy as np
import math

import os, sys
import pickle

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go

import lifelines
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

import tensorflow as tf
import tensorflow_probability as tfp

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, layers

import scipy.stats as stats
from scipy.stats import norm, t, probplot, pearsonr, spearmanr, rankdata
from scipy.stats import truncnorm as truncnorm_scipy
from scipy.stats import gamma as gamma_dist
from scipy.special import gamma

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

import thetaflow as thf

import json
import gc
import glob
from pathlib import Path

sys.path.append("../")
import gndr_utils as utils

import pyarrow.parquet as pq

# Access metadata without loading data
# metadata_train = pq.read_metadata('../Cross Validation Data/train_data_small.parquet')
# metadata_test = pq.read_metadata('../Cross Validation Data/test_data_small.parquet')
metadata_train = pq.read_metadata('../train_data.parquet')
metadata_val = pq.read_metadata('../val_data.parquet')
metadata_test = pq.read_metadata('../test_data.parquet')

# Get shape: (rows, columns)
num_rows_train = metadata_train.num_rows
num_cols_train = metadata_train.num_columns
num_rows_val = metadata_val.num_rows
num_rows_test = metadata_test.num_rows

# The number of features is given by the total number of columns minus the time and event indicator columns
n_features = num_cols_train - 2

n_train = num_rows_train
n_val = num_rows_val
n_test = num_rows_test

print("{} features".format(n_features))
print("{} Linhas - Treino".format(n_train))
print("{} Linhas - Validação".format(n_val))
print("{} Linhas - Teste".format(n_test))

# Parquet file path
parquet_file_path_train = "../train_data.parquet"
parquet_file_path_val = "../val_data.parquet"
parquet_file_path_test = "../test_data.parquet"

parquet_reader_train = pq.ParquetFile(parquet_file_path_train)
parquet_reader_val = pq.ParquetFile(parquet_file_path_val)
parquet_reader_test = pq.ParquetFile(parquet_file_path_test)

def tabular_batch_generator_train(time_col, event_col, batch_size):
    time_col = time_col.decode("utf-8")
    event_col = event_col.decode("utf-8")
    
    # Pull data from parquet file iteratively in chunks
    for batch in parquet_reader_train.iter_batches(batch_size = batch_size):
        # Convert table to pandas DataFrame for column sclicing
        df = batch.to_pandas()

        # Obtain the time and censorship information from the dataset
        time = df[time_col].values.reshape(-1, 1)
        event = df[event_col].values.reshape(-1, 1)
        # Remove the response variable values from the table
        X = df.drop(columns = [time_col, event_col])
        
        indices_z = [0, 4, 5, 6, 7, 8, 9] 
        indices_x = [i for i in range(n_features) if i not in indices_z]

        z = tf.gather(X, indices_z, axis = 1)
        x = tf.gather(X, indices_x, axis = 1)
        
        # Yield the exact tuple structure thetaflow expects: (X, time, event) as a generator
        yield (x, z, time, event)

def tabular_batch_generator_val(time_col, event_col, batch_size):
    time_col = time_col.decode("utf-8")
    event_col = event_col.decode("utf-8")
    
    # Pull data from parquet file iteratively in chunks
    for batch in parquet_reader_val.iter_batches(batch_size = batch_size):
        # Convert table to pandas DataFrame for column sclicing
        df = batch.to_pandas()

        # Obtain the time and censorship information from the dataset
        time = df[time_col].values.reshape(-1, 1)
        event = df[event_col].values.reshape(-1, 1)
        # Remove the response variable values from the table
        X = df.drop(columns = [time_col, event_col])
        
        indices_z = [0, 4, 5, 6, 7, 8, 9] 
        indices_x = [i for i in range(n_features) if i not in indices_z]

        z = tf.gather(X, indices_z, axis = 1)
        x = tf.gather(X, indices_x, axis = 1)
        
        # Yield the exact tuple structure thetaflow expects: (X, time, event) as a generator
        yield (x, z, time, event)

def tabular_batch_generator_test(time_col, event_col, batch_size):
    time_col = time_col.decode("utf-8")
    event_col = event_col.decode("utf-8")
    
    # Pull data from parquet file iteratively in chunks
    for batch in parquet_reader_test.iter_batches(batch_size = batch_size):
        # Convert table to pandas DataFrame for column sclicing
        df = batch.to_pandas()

        # Obtain the time and censorship information from the dataset
        time = df[time_col].values.reshape(-1, 1)
        event = df[event_col].values.reshape(-1, 1)
        # Remove the response variable values from the table
        X = df.drop(columns = [time_col, event_col])

        indices_z = [0, 4, 5, 6, 7, 8, 9] 
        indices_x = [i for i in range(n_features) if i not in indices_z]

        z = tf.gather(X, indices_z, axis = 1)
        x = tf.gather(X, indices_x, axis = 1)
        
        # Yield the exact tuple structure thetaflow expects: (X, time, event) as a generator
        yield (x, z, time, event)


train_batch_size = 350000
train_ds = tf.data.Dataset.from_generator(
    tabular_batch_generator_train,
    args=("tempo", "delta", train_batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, n_features-7), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 7), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype = tf.float32)
    )
)

num_batches_train = int(np.ceil( n_train / train_batch_size ))

# 3. Apply the transformation and keep the GPU fed
train_ds = (
    train_ds
    .apply(tf.data.experimental.assert_cardinality(num_batches_train))
    .prefetch(tf.data.AUTOTUNE)
)



val_batch_size = 350000
val_ds = tf.data.Dataset.from_generator(
    tabular_batch_generator_val,
    args=("tempo", "delta", val_batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, n_features-7), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 7), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype = tf.float32)
    )
)

num_batches_val = int(np.ceil( n_val / val_batch_size ))

# 3. Apply the transformation and keep the GPU fed
val_ds = (
    val_ds
    .apply(tf.data.experimental.assert_cardinality(num_batches_val))
    .prefetch(tf.data.AUTOTUNE)
)


test_batch_size = 350000
test_ds = tf.data.Dataset.from_generator(
    tabular_batch_generator_test,
    args=("tempo", "delta", test_batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, n_features-7), dtype = tf.float32),
        tf.TensorSpec(shape=(None, 7), dtype = tf.float32),
        tf.TensorSpec(shape=(None,1), dtype = tf.float32),
        tf.TensorSpec(shape=(None,1), dtype = tf.float32)
    )
)

num_batches_test = int(np.ceil( n_test / test_batch_size ))

# Keep the GPU fed during optimization
test_ds = (
    test_ds
    .apply( tf.data.experimental.assert_cardinality(num_batches_test) )
    .prefetch(tf.data.AUTOTUNE)
)

def build_lognormal_model_linear_effects():
    def softplus_inv(u):
        return tf.math.log(tf.math.exp(u) - 1)

    parameters = {
        "beta_mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": 7, "init": 1.0},
        "mu_nn": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": 1.7899947},
        "sigma": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": 1.5657033},
    }

    def loglikelihood_loss(model, nn_output, data):
        X, z_mu, y, delta = data
        eps = 1e-7
        y_safe = y + eps
        
        beta_mu = model.get_variable("beta_mu")[:,None]
        mu_nn = model.get_variable("mu_nn", nn_output)
        sigma = model.get_variable("sigma", nn_output)

        mu = tf.math.softplus( tf.matmul(z_mu, beta_mu) + mu_nn )
        
        log_y = tf.math.log(y + 1.0e-7)
        z = (log_y - mu) / sigma

        S_y = 0.5 * tf.math.erfc(z / tf.math.sqrt(2.0))
        log_S_y = tf.math.log(S_y + 1.0e-7)
        
        neg_loglik = -tf.reduce_sum( delta * (-log_y - tf.math.log(sigma) - z**2.0 / 2.0 ) + (1.0-delta) * log_S_y )

        return neg_loglik

    def neural_network(model, seed = None):
        initializer = tf.keras.initializers.GlorotNormal(seed=seed)
        
        model.dense1 = layers.Dense(
            units = 128, 
            activation = "gelu",
            kernel_initializer = initializer,
            name = "tabular_features_extractor"
        )
        model.dense2 = layers.Dense(
            units = 64,
            activation = "gelu",
            kernel_initializer = initializer,
            name = "interaction_layer_1"
        )
        model.dense3 = layers.Dense(
            units = 32,
            activation = "gelu",
            kernel_initializer = initializer,
            name = "interaction_layer_2"
        )
        model.dense4 = layers.Dense(
            units = 8,
            activation = "gelu",
            kernel_initializer = initializer,
            name = "interaction_layer_2"
        )
        model.output_layer = layers.Dense(
            units = 2,
            activation = None, # Linear, o exponente fica na Loss function
            kernel_initializer = initializer,
            name = "log_lambda_output"
        )
    
    def neural_network_call(model, x_input, training = False):
        x = model.dense1(x_input)
        x = model.dense2(x)
        x = model.dense3(x)
        x = model.output_layer(x)
        return x
    
    def neural_network_call_nolast(model, x_input):
        x = model.dense1(x_input)
        x = model.dense2(x)
        x = model.dense3(x)
        return x

    return parameters, loglikelihood_loss, neural_network, neural_network_call, neural_network_call_nolast



with tf.device("/GPU:0"):
    lognormal_parameters, lognormal_loss, lognormal_neural_network, lognormal_call, lognormal_call_nolast = \
    build_lognormal_model_linear_effects( )
    seed = 10
    lognormal_model_linear = thf.ModelNN(lognormal_parameters, lognormal_loss,
                                       lognormal_neural_network, lognormal_call,
                                       lognormal_call_nolast, input_dim = (87,), seed = seed)

    lognormal_model_linear.load_model("lognormal")
    
    lognormal_model_linear.pre_train_model(epochs = None, x = train_ds, data = None, n_train = n_train, shuffle = True)
    lognormal_model_linear.train_model(epochs = 50, x = train_ds, data = None, n_train = n_train,
                                     shuffle = True,
                                     get_covariances = True,
                                     validation = True, x_val = val_ds, data_val = None, n_val = n_val,
                                     force_training_validation = False,
                                     optimizer_independent = optimizers.Adam(learning_rate = 0.001, clipnorm = 1.0),
                                     optimizer_nn = optimizers.Adam(learning_rate = 0.001, clipnorm = 1.0),
                                     fine_tune_nn_lr = 0.001, fine_tune_independent_lr = 0.001,
                                     early_stopping = True, early_stopping_patience = 30, 
                                     early_stopping_warmup = 10,
                                     reduce_lr = True, reduce_lr_warmup = 0,
                                     reduce_lr_factor = 0.5, reduce_lr_min_delta = 100, reduce_lr_patience = 25,
                                     reduce_lr_cooldown = 10, reduce_lr_min_lr = 1.0e-5,
                                     fine_tune = True,
                                     finetune_early_stopping = True, finetune_early_stopping_patience = 30,
                                     finetune_early_stopping_warmup = 10,
                                     finetune_reduce_lr = True, finetune_reduce_lr_warmup = 0,
                                     finetune_reduce_lr_factor = 0.5, finetune_reduce_lr_min_delta = 10, finetune_reduce_lr_patience = 25,
                                     finetune_reduce_lr_cooldown = 10, finetune_reduce_lr_min_lr = 1.0e-5,
                                     deterministic = True,
                                     verbose = True, print_freq = 1,
                                     train_batch_size = None, val_batch_size = None,
                                     buffer_size = None, gradient_accumulation_steps = None)

lognormal_model_linear.hessian_jitter = 1.0e-6
lognormal_model_linear.save_model( "lognormal_final" )