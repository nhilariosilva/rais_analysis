import os
import warnings
import time

import pandas as pd
import numpy as np
import math

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
# import modelnn2 as thf

import json
import gc
import glob
from pathlib import Path

import pyarrow.parquet as pq

# Access metadata without loading data
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
print("{} colunas - Treino".format(n_train))
print("{} colunas - Validação".format(n_val))
print("{} colunas - Teste".format(n_test))

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
        
        # Yield the exact tuple structure thetaflow expects: (X, time, event) as a generator
        yield (X, time, event)

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
        
        # Yield the exact tuple structure thetaflow expects: (X, time, event) as a generator
        yield (X, time, event)

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
        
        # Yield the exact tuple structure thetaflow expects: (X, time, event) as a generator
        yield (X, time, event)

train_batch_size = 150000
train_ds = tf.data.Dataset.from_generator(
    tabular_batch_generator_train,
    args=("tempo", "delta", train_batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, n_features), dtype = tf.float32),
        tf.TensorSpec(shape=(None,1), dtype = tf.float32),
        tf.TensorSpec(shape=(None,1), dtype = tf.float32)
    )
)

num_batches_train = int(np.ceil( n_train / train_batch_size ))

# Keep the GPU fed during optimization
train_ds = (
    train_ds
    .apply( tf.data.experimental.assert_cardinality(num_batches_train) )
    .prefetch(tf.data.AUTOTUNE)
)

val_batch_size = 150000
val_ds = tf.data.Dataset.from_generator(
    tabular_batch_generator_val,
    args=("tempo", "delta", val_batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, n_features), dtype = tf.float32),
        tf.TensorSpec(shape=(None,1), dtype = tf.float32),
        tf.TensorSpec(shape=(None,1), dtype = tf.float32)
    )
)

num_batches_val = int(np.ceil( n_val / val_batch_size ))

# Keep the GPU fed during optimization
val_ds = (
    val_ds
    .apply( tf.data.experimental.assert_cardinality(num_batches_val) )
    .prefetch(tf.data.AUTOTUNE)
)


test_batch_size = 150000
test_ds = tf.data.Dataset.from_generator(
    tabular_batch_generator_test,
    args=("tempo", "delta", test_batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, n_features), dtype = tf.float32),
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

def build_mixture_weibull_model( ridge_penalty = 1e-4, lasso_penalty = 1e-4 ):
    lambda_max = 40.0
    
    def softplus_inv(y):
        return tf.math.log(tf.math.exp(y) - 1)
        
    def scaled_sigmoid_link(x):
        '''
            Smoothly bounds the neural network output between 0 and lambda_max.
        '''
        return lambda_max * tf.math.sigmoid(x)
    
    def scaled_sigmoid_link_inv(y):
        '''
            The inverse of the scaled sigmoid (the scaled logit function).
            Includes epsilon clipping for numerical stability.
        '''
        # Normalize the value back to the (0, 1) range
        u = y / lambda_max
        # Clip u to be strictly between 0 and 1 to avoid log(0) or division by zero
        u = tf.clip_by_value(u, 1.0e-7, 1.0 - 1.0e-7)
        # Apply the logit transformation
        return -( tf.math.log(1.0 - u) - tf.math.log(u) )

    def logit(u):
        return -( tf.math.log(1-u) - tf.math.log(u) )
    
    # Shape parameter: constant
    # Scale parameter: neural network
    # Cure probability: neural network
    mixture_weibull_parameters = {
        "k": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": 1.0, "warmup_time": 0},
        "lam": {"link": scaled_sigmoid_link, "link_inv": scaled_sigmoid_link_inv, "par_type": "nn", "shape": 1, "init": 1.0, "warmup_time": 0},
        "p": {"link": tf.math.sigmoid, "link_inv": logit, "par_type": "nn", "shape": 1, "init": 0.5, "warmup_time": 0}
    }

    def loglikelihood_loss(model, nn_output, data):
        X, y, delta = data
        
        k = model.get_variable("k")
        # k = model.get_variable("k", nn_output)
        lam = model.get_variable("lam", nn_output)
        p = model.get_variable("p", nn_output)

        log_y = tf.math.log(y + 1.0e-7)
        log_lam = tf.math.log(lam + 1.0e-7)
        log_S0 = -( y / (lam + 1.0e-7) )**k 
        S0 = tf.math.exp( log_S0 )
        
        log_f0 = tf.math.log(k + 1.0e-7) - k * log_lam + (k - 1.0) * log_y + log_S0
        
        loglik_terms = delta * (tf.math.log(1.0 - p + 1.0e-7) + log_f0) + \
                       (1.0 - delta) * tf.math.log(p + (1.0 - p)*S0 + 1.0e-7)
        neg_loglik = -tf.reduce_sum(loglik_terms)
        
        return neg_loglik

    def neural_network(model, input_dim=100, seed=None):
        initializer = tf.keras.initializers.GlorotNormal(seed=seed)
        
        elastic_net = tf.keras.regularizers.L1L2(l1 = lasso_penalty, l2 = ridge_penalty)
        
        model.dense1 = layers.Dense(
            units = 128, 
            activation = "gelu",
            kernel_initializer = initializer,
            kernel_regularizer = elastic_net,
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

    return mixture_weibull_parameters, loglikelihood_loss, neural_network, neural_network_call, neural_network_call_nolast

with tf.device("/GPU:0"):
    mixture_weibull_parameters, mixture_weibull_loss, mixture_weibull_neural_network, mixture_weibull_call, mixture_weibull_call_nolast = \
    build_mixture_weibull_model( ridge_penalty = 0.001, lasso_penalty = 0.001 )
    seed = 10
    mixture_weibull_model = thf.ModelNN(mixture_weibull_parameters, mixture_weibull_loss,
                                        mixture_weibull_neural_network, mixture_weibull_call,
                                        mixture_weibull_call_nolast, input_dim = (n_features,), seed = seed)
    mixture_weibull_model.pre_train_model(epochs = None, x = train_ds, data = None, n_train = n_train, shuffle = True)
    mixture_weibull_model.train_model(epochs = 1500, x = train_ds, data = None, n_train = n_train,
                                      shuffle = True,
                                      get_covariances = True,
                                      validation = True, x_val = val_ds, n_val = n_val,
                                      force_training_validation = False,
                                      optimizer_independent = optimizers.Adam(learning_rate = 0.0001, clipnorm = 1.0),
                                      optimizer_nn = optimizers.Adam(learning_rate = 0.0001, clipnorm = 1.0),
                                      fine_tune_nn_lr = 0.0001, fine_tune_independent_lr = 0.0001,
                                      early_stopping = True, early_stopping_patience = 50, 
                                      early_stopping_warmup = 10,
                                      reduce_lr = True, reduce_lr_warmup = 0,
                                      reduce_lr_factor = 0.5, reduce_lr_min_delta = 1.0e-3, reduce_lr_patience = 25,
                                      reduce_lr_cooldown = 10, reduce_lr_min_lr = 1.0e-5,
                                      fine_tune = True,
                                      finetune_early_stopping = True, finetune_early_stopping_patience = 50,
                                      finetune_early_stopping_warmup = 10,
                                      finetune_reduce_lr = True, finetune_reduce_lr_warmup = 0,
                                      finetune_reduce_lr_factor = 0.5, finetune_reduce_lr_min_delta = 1.0e-2, finetune_reduce_lr_patience = 25,
                                      finetune_reduce_lr_cooldown = 10, finetune_reduce_lr_min_lr = 1.0e-5,
                                      deterministic = True,
                                      verbose = True, print_freq = 1,
                                      train_batch_size = None, val_batch_size = None,
                                      buffer_size = None, gradient_accumulation_steps = 1)

mixture_weibull_model.hessian_jitter = 1.0e-6
mixture_weibull_model.save_model("mixture_weibull_model")