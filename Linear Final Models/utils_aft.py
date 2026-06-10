import os, sys
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

import json
import gc
import glob
from pathlib import Path


def get_simple_model( dist = "weibull", theta_init = None ):
    '''
        Return a simple model structure considering no covariates. Just the target distribution with independent parameters

        dist: Name of the distribution user wants to fit: ["exponential", "weibull", "lognormal", "loglogistic", "bs", "kwcwg"]
    '''
    dist = dist.lower()
    if(dist == "exponential"):
        parameters = {
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0}
        }
        
        def loglikelihood_loss(model, nn_output, data):
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype=tf.float32)
        
            scale = model.get_variable("scale")
        
            log_h_base = -tf.math.log(scale + eps)
            log_S_base = -y / scale
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_mean(loglik_terms)

    elif(dist == "weibull"):
        parameters = {
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            k = model.get_variable("shape")
            lam = model.get_variable("scale")
        
            log_h_base = tf.math.log(k) - k * tf.math.log(lam) + (k-1) * tf.math.log( y + eps )
            log_S_base = - (y / lam)**k
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_mean(loglik_terms)

    elif(dist == "lognormal"):
        parameters = {
            "mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            mu = model.get_variable("mu")
            scale = model.get_variable("scale")

            log_y = tf.math.log(y + eps)
            
            normal_dist = tfp.distributions.Normal(loc = mu, scale = scale)
            
            # Y ~ Lognormal(mu, sigma) => X = log(Y) ~ N(mu, sigma²)
            # P(Y > y) = P(log(Y) > log(y)) = P(X > log(y)) = S_N( log(y) )
            log_S_base = normal_dist.log_survival_function( log_y )
            # P(Y <= y) = Phi( (log(y) - mu) / sigma ) 
            # Therefore, f_Y(y) = phi( (log(y) - mu)/sigma ) * (y sigma)^(-1) => f_Y(y) = f_X( log_y ) / y
            log_f_base = normal_dist.log_prob(log_y) - log_y
            log_h_base = log_f_base - log_S_base
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    elif(dist == "loglogistic"):
        parameters = {
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("scale")
            b = model.get_variable("shape")

            log_y = tf.math.log(y + eps)
            
            log_b = tf.math.log(b)
            log_a = tf.math.log(a)

            log_terms = tf.math.softplus( b*( log_y - log_a ) )
            log_S_base = -log_terms
            log_h_base = log_b - b * log_a + (b-1) * log_y - log_terms
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    elif( dist == "bs" ):        
        parameters = {
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("shape")
            b = model.get_variable("scale")
            
            pi = tf.constant( 3.141592653589793, dtype = tf.float32 )
            log_b = tf.math.log(b)
            log_a = tf.math.log(a)

            sqrt_b_y = tf.math.sqrt( b / (y+eps) )
            sqrt_y_b = tf.math.sqrt( y / b )
            normal_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
            z_score = (sqrt_y_b - sqrt_b_y) / a
            log_S_base = normal_dist.log_survival_function(z_score)
            log_f_base = -tf.math.log(2*tf.math.sqrt(2 * pi)) - log_a - log_b + \
                          tf.math.log( sqrt_b_y + tf.math.pow(sqrt_b_y, 3) ) - (y / b + b / (y + eps) - 2) / (2*a**2)
            log_h_base = log_f_base - log_S_base
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    elif( dist == "kwcwg" ):
        def logit(u):
            return -( tf.math.log(1-u) - tf.math.log(u) )
        
        parameters = {
            "a": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "b": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "alpha": {"link": tf.math.sigmoid, "link_inv": logit, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "gamma": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0},
            "lam": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("a")
            b = model.get_variable("b")
            alpha = model.get_variable("alpha")
            gamma = model.get_variable("gamma")
            lam = model.get_variable("lam")
            
            log_y = tf.math.log(y + eps)

            alpha = tf.clip_by_value(alpha, 1e-5, 1-1e-5)
            log_a = tf.math.log(a)
            log_b = tf.math.log(b)
            log_alpha = tf.math.log(alpha)
            log_gamma = tf.math.log(gamma)
            log_lam = tf.math.log(lam)

            lambdat_gamma = (lam * y)**gamma
            exp_lambdat_gamma = tf.math.exp( -lambdat_gamma )
            log_S_term = tf.math.log( 1 - ( alpha*(1-exp_lambdat_gamma) / (alpha + (1-alpha)*exp_lambdat_gamma) )**a )
            
            log_f_base = a*log_alpha + log_gamma + log_a + log_b + gamma*log_lam + (gamma-1)*log_y \
                         - lambdat_gamma + (a-1) * tf.math.log( 1 - exp_lambdat_gamma ) \
                         - (a+1) * tf.math.log( alpha + (1-alpha)*exp_lambdat_gamma ) \
                         + (b-1) * log_S_term
            log_S_base = b * log_S_term
            log_h_base = log_f_base - log_S_base
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    else:
        raise Exception("Error: Distribution {} is not available".format(dist))
        

    # Cycle through all parameters and set the initial value corresponding to a zero unconstrained value
    for par_name in parameters:
        # If initial theta was not given, consider the initial value to be zero on the unconstrained scale
        if(theta_init is None or par_name not in theta_init):
            parameters[par_name]["init"] = parameters[par_name]["link"]( 0.0 )
        # Otherwise, just set the given values to each parameter
        else:
            parameters[par_name]["init"] = theta_init[par_name]

    return parameters, loglikelihood_loss


def get_AFT_model( X, dist = "weibull", theta_init = None, beta_init = None ):
    '''
        Given a design matrix for the variables we have the interest of interpreting and the desired AFT
        distribution, we create the corresponding thetaflow native function definitions of that model, considering the
        vector of linear coefficients to be the exact same size as X.

        X: Design matrix for interpretable data
        dist: Name of the distribution user wants to fit: ["exponential", "weibull", "lognormal", "loglogistic", "bs", "kwcwg"]
        theta_init: Vector of initial values for base distribution parameters. It expects a dict whose keys
                    correspond to the specific distribution chosen.
                    List of parameter names:
                        - exponential: scale
                        - weibull: shape, scale
                        - lognormal: mu, scale
                        - loglogistic: shape, scale
                        - bs (Birnbaum-Saunders): shape, scale
                        - kwcwg (Kumaraswamy complementary Weibull Geometric): a, b, alpha, gamma, lam
    '''
    dist = dist.lower()
    if(dist == "exponential"):
        nn_output_size = 1

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)

        # parameters = {
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        # }
        
        parameters = {
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        }
        
        def loglikelihood_loss(model, nn_output, data):
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype=tf.float32)
        
            scale = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]
        
            # Linear predictor and acceleration in time
            r_z = tf.matmul(z, beta)
            t0 = y * tf.math.exp(-r_z)
        
            # Basal evaluation at t0
            log_h_base = -tf.math.log(scale + eps)
            log_S_base = -t0 / scale
    
            # Evaluation of h_AFT
            log_h_AFT = log_h_base - r_z
            
            loglik_terms = delta * log_h_AFT + log_S_base
            return -tf.reduce_mean(loglik_terms)
            # return -tf.reduce_sum(loglik_terms)
            
    elif(dist == "weibull"):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        # parameters = {
        #     "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        # }

        parameters = {
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            k = model.get_variable("shape", nn_output)
            lam = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]
            
            # Linear predictor and acceleration in time
            r_z = tf.matmul(z, beta)
            t0 = y * tf.math.exp(-r_z)
        
            # Basal evaluation at t0
            log_h_base = tf.math.log(k) - k * tf.math.log(lam) + (k-1) * tf.math.log( t0 + eps )
            log_S_base = - (t0 / lam)**k
    
            # Evaluation of h_AFT
            log_h_AFT = log_h_base - r_z
            
            loglik_terms = delta * log_h_AFT + log_S_base
            return -tf.reduce_mean(loglik_terms)

    elif(dist == "lognormal"):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        # parameters = {
        #     "mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        # }

        parameters = {
            "mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            mu = model.get_variable("mu", nn_output)
            scale = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]

            # Linear predictor and acceleration in time
            r_z = tf.matmul(z, beta)
            t0 = y * tf.math.exp(-r_z)
            log_t0 = tf.math.log(t0 + eps)
            
            normal_dist = tfp.distributions.Normal(loc = mu, scale = scale)
            
            # Basal evaluation at t0
            # Y ~ Lognormal(mu, sigma) => X = log(Y) ~ N(mu, sigma²)
            # P(Y > y) = P(log(Y) > log(y)) = P(X > log(y)) = S_N( log(y) )
            log_S_base = normal_dist.log_survival_function( log_t0 )
            # P(Y <= y) = Phi( (log(y) - mu) / sigma ) 
            # Therefore, f_Y(y) = phi( (log(y) - mu)/sigma ) * (y sigma)^(-1) => f_Y(y) = f_X( log_y ) / y
            log_f_base = normal_dist.log_prob(log_t0) - log_t0
            log_h_base = log_f_base - log_S_base
            
            # Evaluation of h_AFT
            log_h_AFT = log_h_base - r_z
            
            loglik_terms = delta * log_h_AFT + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    elif(dist == "loglogistic"):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)

        # parameters = {
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        # }
            
        parameters = {
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("scale", nn_output)
            b = model.get_variable("shape", nn_output)
            beta = model.get_variable("beta")[:,None]

            # Linear predictor and acceleration in time
            r_z = tf.matmul(z, beta)
            t0 = y * tf.math.exp(-r_z)
            log_t0 = tf.math.log(t0 + eps)
            
            log_b = tf.math.log(b)
            log_a = tf.math.log(a)

            log_terms = tf.math.softplus( b*( log_t0 - log_a ) )
            log_S_base = -log_terms
            log_h_base = log_b - b * log_a + (b-1) * log_t0 - log_terms
            
            # Evaluation of h_AFT
            log_h_AFT = log_h_base - r_z
            
            loglik_terms = delta * log_h_AFT + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    elif( dist == "bs" ):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        # parameters = {
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        # }

        parameters = {
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("shape", nn_output)
            b = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]

            # Linear predictor and acceleration in time
            r_z = tf.matmul(z, beta)
            t0 = y * tf.math.exp(-r_z)
            
            pi = tf.constant( 3.141592653589793, dtype = tf.float32 )
            log_b = tf.math.log(b)
            log_a = tf.math.log(a)

            sqrt_b_t0 = tf.math.sqrt( b / (t0+eps) )
            sqrt_t0_b = tf.math.sqrt( t0 / b )
            normal_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
            z_score = (sqrt_t0_b - sqrt_b_t0) / a
            log_S_base = normal_dist.log_survival_function(z_score)
            log_f_base = -tf.math.log(2*tf.math.sqrt(2 * pi)) - log_a - log_b + \
                          tf.math.log( sqrt_b_t0 + tf.math.pow(sqrt_b_t0, 3) ) - (t0 / b + b / (t0 + eps) - 2) / (2*a**2)
            log_h_base = log_f_base - log_S_base
            
            # Evaluation of h_AFT
            log_h_AFT = log_h_base - r_z
            
            loglik_terms = delta * log_h_AFT + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    elif( dist == "kwcwg" ):
        nn_output_size = 5

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        def logit(u):
            return -( tf.math.log(1-u) - tf.math.log(u) )
        
        # parameters = {
        #     "a": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "b": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "alpha": {"link": tf.math.sigmoid, "link_inv": logit, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "gamma": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "lam": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        # }

        parameters = {
            "a": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "b": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "alpha": {"link": tf.math.sigmoid, "link_inv": logit, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "gamma": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "lam": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": 0}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("a", nn_output)
            b = model.get_variable("b", nn_output)
            alpha = model.get_variable("alpha", nn_output)
            gamma = model.get_variable("gamma", nn_output)
            lam = model.get_variable("lam", nn_output)
            beta = model.get_variable("beta")[:,None]
            
            # Linear predictor and acceleration in time
            r_z = tf.matmul(z, beta)
            t0 = y * tf.math.exp(-r_z)
            log_t0 = tf.math.log(t0 + eps)

            alpha = tf.clip_by_value(alpha, 1e-5, 1-1e-5)
            log_a = tf.math.log(a)
            log_b = tf.math.log(b)
            log_alpha = tf.math.log(alpha)
            log_gamma = tf.math.log(gamma)
            log_lam = tf.math.log(lam)

            lambdat_gamma = (lam * t0)**gamma
            exp_lambdat_gamma = tf.math.exp( -lambdat_gamma )
            log_S_term = tf.math.log( 1 - ( alpha*(1-exp_lambdat_gamma) / (alpha + (1-alpha)*exp_lambdat_gamma) )**a )
            
            log_f_base = a*log_alpha + log_gamma + log_a + log_b + gamma*log_lam + (gamma-1)*log_t0 \
                         - lambdat_gamma + (a-1) * tf.math.log( 1 - exp_lambdat_gamma ) \
                         - (a+1) * tf.math.log( alpha + (1-alpha)*exp_lambdat_gamma ) \
                         + (b-1) * log_S_term
            log_S_base = b * log_S_term
            log_h_base = log_f_base - log_S_base
            
            # Evaluation of h_AFT
            log_h_AFT = log_h_base - r_z
            
            loglik_terms = delta * log_h_AFT + log_S_base
            return -tf.reduce_mean(loglik_terms)
            
    else:
        raise Exception("Error: Distribution {} is not available".format(dist))
        

    # Cycle through all parameters and set the initial value corresponding to a zero unconstrained value
    for par_name in parameters:
        if(par_name != "beta"):
            # If initial theta was not given, consider the initial value to be zero on the unconstrained scale
            if(theta_init is None or par_name not in theta_init):
                parameters[par_name]["init"] = parameters[par_name]["link"]( 0.0 )
            # Otherwise, just set the given values to each parameter
            else:
                parameters[par_name]["init"] = theta_init[par_name]

    # If beta init was not given, simply set all coefficients to zeros
    if(beta_init is None):
        parameters["beta"]["init"] = np.repeat(0.0, X.shape[1])
    else:
        parameters["beta"]["init"] = beta_init

    return parameters, loglikelihood_loss, nn_output_size


def build_AFT_model( Z, dist  = "weibull", theta_init = None, beta_init = None ):

    parameters, loglikelihood_loss, nn_output_size = get_AFT_model( Z, dist = dist, theta_init = theta_init, beta_init = beta_init )
    
    def neural_network(model, seed = None):
        initializer = tf.keras.initializers.HeNormal(seed = seed)

        model.dense1 = layers.Dense(
            units = 128, 
            activation = "gelu",
            kernel_initializer = initializer
        )
        model.dense2 = layers.Dense(
            units = 64,
            activation = "gelu",
            kernel_initializer = initializer
        )
        model.dense3 = layers.Dense(
            units = 32,
            activation = "gelu",
            kernel_initializer = initializer
        )
        model.dense4 = layers.Dense(
            units = 8,
            activation = "gelu",
            kernel_initializer = initializer
        )
        model.output_layer = layers.Dense(
            units = nn_output_size,
            activation = None, # Linear, o exponente fica na Loss function
            use_bias = True,
            kernel_initializer = tf.keras.initializers.Zeros()
        )
    
    def neural_network_call(model, x_input, training = False):
        x = model.dense1(x_input)
        x = model.dense2(x)
        x = model.dense3(x)
        x = model.dense4(x)
        x = model.output_layer(x)
        return x
    
    def neural_network_call_nolast(model, x_input):
        x = model.dense1(x_input)
        x = model.dense2(x)
        x = model.dense3(x)
        x = model.dense4(x)
        return x

    return parameters, loglikelihood_loss, neural_network, neural_network_call, neural_network_call_nolast



def get_survival_exp(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100):
    scale_train = model.predict(X_train)["scale"].numpy().flatten()
    scale_test = model.predict(X_test)["scale"].numpy().flatten()
    beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

    r_z_train = np.dot(z_train, beta).flatten()
    t0_grid_train = ts_grid * np.exp( -r_z_train )
    S_ts_train = np.exp( - t0_grid_train / scale_train )
    t0_train = y_train * np.exp( -r_z_train )
    S_train = np.exp( - t0_train / scale_train )
    H_train = -np.log( S_train )

    r_z_test = np.dot(z_test, beta).flatten()
    t0_grid_test = ts_grid * np.exp( -r_z_test )
    S_ts_test = np.exp( -t0_grid_test / scale_test )
    t0_test = y_test * np.exp( -r_z_test )
    S_test = np.exp( - t0_test / scale_test )
    H_test = -np.log( S_test )

    return {
        "ts_grid": ts_grid,
        "S_ts_train": S_ts_train,
        "S_ts_test": S_ts_test,
        "S_train": S_train,
        "S_test": S_test,
        "H_train": H_train,
        "H_test": H_test
    }

def get_survival_weibull(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    k_train = pred_train["shape"].numpy().flatten()
    lam_train = pred_train["scale"].numpy().flatten()
    k_test = pred_test["shape"].numpy().flatten()
    lam_test = pred_test["scale"].numpy().flatten()
    beta = model.predict("beta")[:,None]

    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    
    r_z_train = np.dot(z_train, beta).flatten()
    t0_grid_train = ts_grid * np.exp( -r_z_train )
    S_ts_train = np.exp( - (t0_grid_train / lam_train)**k_train )
    t0_train = y_train * np.exp( -r_z_train )
    S_train = np.exp( - (t0_train / lam_train)**k_train )
    H_train = -np.log( S_train )
    
    r_z_test = np.dot(z_test, beta).flatten()
    t0_grid_test = ts_grid * np.exp( -r_z_test )
    S_ts_test = np.exp( - (t0_grid_test / lam_test)**k_test )
    t0_test = y_test * np.exp( -r_z_test )
    S_test = np.exp( - (t0_test / lam_test)**k_test )
    H_test = -np.log( S_test )
    
    S0_ts_test = np.exp( -(ts_grid / lam_test)**k_test )
    S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).T )
    S_test = np.exp( - y_test / lam_test )
    H_test = -np.log( S_test )

    return {
        "ts_grid": ts_grid,
        "S_ts_train": S_ts_train,
        "S_ts_test": S_ts_test,
        "S_train": S_train,
        "S_test": S_test,
        "H_train": H_train,
        "H_test": H_test
    }

def get_survival_lognormal(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    mu_train = pred_train["mu"].numpy().flatten()
    sigma_train = pred_train["scale"].numpy().flatten()
    mu_test = pred_test["mu"].numpy().flatten()
    sigma_test = pred_test["scale"].numpy().flatten()
    beta = model.predict("beta")[:,None]

    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    
    r_z_train = np.dot(z_train, beta).flatten()
    t0_grid_train = ts_grid * np.exp( -r_z_train )
    log_t0_grid_train = np.log( t0_grid_train )
    
    normal_dist_train = tfp.distributions.Normal(loc = mu_train, scale = sigma_train)
    S_ts_train = normal_dist_train.survival_function( log_t0_grid_train ).numpy()
    t0_train = y_train * np.exp( -r_z_train )
    log_t0_train = np.log(t0_train)
    S_train = normal_dist_train.survival_function( log_t0_train ).numpy()
    H_train = -np.log( S_train )

    r_z_test = np.dot(z_test, beta).flatten()
    t0_grid_test = ts_grid * np.exp( -r_z_test )
    log_t0_grid_test = np.log( t0_grid_test )
    
    normal_dist_test = tfp.distributions.Normal(loc = mu_test, scale = sigma_test)
    S_ts_test = normal_dist_test.survival_function( log_t0_grid_test ).numpy()
    t0_test = y_test * np.exp( -r_z_test )
    log_t0_test = np.log(t0_test)
    S_test = normal_dist_test.survival_function( log_t0_test ).numpy()
    H_test = -np.log( S_test )
    
    return {
        "ts_grid": ts_grid,
        "S_ts_train": S_ts_train,
        "S_ts_test": S_ts_test,
        "S_train": S_train,
        "S_test": S_test,
        "H_train": H_train,
        "H_test": H_test
    }


def get_survival_loglogistic(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    a_train = pred_train["scale"].numpy().flatten()
    b_train = pred_train["shape"].numpy().flatten()
    a_test = pred_test["scale"].numpy().flatten()
    b_test = pred_test["shape"].numpy().flatten()
    beta = model.predict("beta")[:,None]

    log_a_train = np.log(a_train)
    log_a_test = np.log(a_test)

    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    
    r_z_train = np.dot(z_train, beta).flatten()
    t0_grid_train = ts_grid * np.exp( -r_z_train )
    log_t0_grid_train = np.log( t0_grid_train )
    
    terms_train_t0 = tf.math.softplus( b_train*( log_t0_grid_train - log_a_train ) ).numpy()
    S_ts_train = np.exp( -terms_train_t0 )
    t0_train = y_train * np.exp( -r_z_train )
    log_t0_train = np.log( t0_train )
    terms_train = tf.math.softplus( b_train*( log_t0_train - log_a_train ) ).numpy()
    S_train = np.exp( -terms_train )
    H_train = -np.log( S_train )

    r_z_test = np.dot(z_test, beta).flatten()
    t0_grid_test = ts_grid * np.exp( -r_z_test )
    log_t0_grid_test = np.log( t0_grid_test )
    
    terms_test_t0 = tf.math.softplus( b_test*( log_t0_grid_test - log_a_test ) ).numpy()
    S_ts_test = np.exp( -terms_test_t0 )
    t0_test = y_test * np.exp( -r_z_test )
    log_t0_test = np.log(t0_test)
    terms_test = tf.math.softplus( b_test*( log_t0_test - log_a_test ) ).numpy()
    S_test = np.exp( -terms_test )
    H_test = -np.log( S_test )
    
    return {
        "ts_grid": ts_grid,
        "S_ts_train": S_ts_train,
        "S_ts_test": S_ts_test,
        "S_train": S_train,
        "S_test": S_test,
        "H_train": H_train,
        "H_test": H_test
    }


def get_survival_bs(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100):
    eps = tf.constant(1.0e-7, dtype = tf.float32)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    a_train = pred_train["shape"].numpy().flatten()
    b_train = pred_train["scale"].numpy().flatten()
    a_test = pred_test["shape"].numpy().flatten()
    b_test = pred_test["scale"].numpy().flatten()
    beta = model.predict("beta")[:,None]

    normal_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    
    r_z_train = np.dot(z_train, beta).flatten()
    t0_grid_train = ts_grid * np.exp( -r_z_train )
    
    sqrt_b_t0_grid_train = tf.math.sqrt( b_train / (t0_grid_train+eps) )
    sqrt_t0_grid_b_train = tf.math.sqrt( t0_grid_train / b_train )
    zs_score_train = (sqrt_t0_grid_b_train - sqrt_b_t0_grid_train) / a_train
    S_ts_train = normal_dist.survival_function(zs_score_train).numpy()

    t0_train = y_train * np.exp( -r_z_train )
    
    sqrt_b_t0_train = tf.math.sqrt( b_train / (t0_train+eps) )
    sqrt_t0_b_train = tf.math.sqrt( t0_train / b_train )
    z_score_train = (sqrt_t0_b_train - sqrt_b_t0_train) / a_train
    S_train = normal_dist.survival_function(z_score_train).numpy()
    H_train = -np.log( S_train )

    r_z_test = np.dot(z_test, beta).flatten()
    t0_grid_test = ts_grid * np.exp( -r_z_test )
    
    sqrt_b_t0_grid_test = tf.math.sqrt( b_test / (t0_grid_test+eps) )
    sqrt_t0_grid_b_test = tf.math.sqrt( t0_grid_test / b_test )
    zs_score_test = (sqrt_t0_grid_b_test - sqrt_b_t0_grid_test) / a_test
    S_ts_test = normal_dist.survival_function(zs_score_test).numpy()

    t0_test = y_test * np.exp( -r_z_test )
    
    sqrt_b_t0_test = tf.math.sqrt( b_test / (t0_test+eps) )
    sqrt_t0_b_test = tf.math.sqrt( t0_test / b_test )
    z_score_test = (sqrt_t0_b_test - sqrt_b_t0_test) / a_test
    S_test = normal_dist.survival_function(z_score_test).numpy()
    H_test = -np.log( S_test )
    
    return {
        "ts_grid": ts_grid,
        "S_ts_train": S_ts_train,
        "S_ts_test": S_ts_test,
        "S_train": S_train,
        "S_test": S_test,
        "H_train": H_train,
        "H_test": H_test
    }

def summary_betas(model, colnames):
    beta_summary = model.summary()

    beta_hats = []
    beta_ses = []
    beta_statistics = []
    beta_pvalues = []
    beta_CIs = []
    for j, beta in enumerate(beta_summary.columns[1::4]):
        beta_hat = float( beta_summary.iloc[0,j*4+1] )
        se = float( beta_summary.iloc[0,j*4+2] )
    
        test_statistic = beta_hat / se
        p_value = 2.0 * (1.0 - norm.cdf(np.abs(test_statistic)))
        
        lower = float( beta_summary.iloc[0,j*4+3] )
        upper = float( beta_summary.iloc[0,j*4+4] )
    
        beta_hats.append( beta_hat )
        beta_ses.append( se )
        beta_statistics.append( test_statistic )
        beta_pvalues.append( p_value )
        beta_CIs.append("({} ; {})".format(np.round(lower,4), np.round(upper,4)))
    beta_summary = pd.DataFrame({"Coef": beta_hats, "Se": beta_ses, "Z": beta_statistics, "pvalue": beta_pvalues, "CI(95%)": beta_CIs})
    beta_summary.index = colnames
    return beta_summary