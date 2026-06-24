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

import bct


def get_simple_model( dist, theta_init = None, warmup_times = None ):
    '''
        Return a simple model structure considering no covariates. Just the target distribution with independent parameters

        dist: Name of the distribution user wants to fit: ["exponential", "weibull", "lognormal", "loglogistic", "bs", "kwcwg"]
    '''
    dist = dist.lower()
    if(dist == "exponential"):
        parameters = {
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
        }
        
        def loglikelihood_loss(model, nn_output, data):
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype=tf.float32)
        
            scale = model.get_variable("scale")
        
            log_h_base = -tf.math.log(scale + eps)
            log_S_base = -y / scale
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_sum(loglik_terms)

    elif(dist == "weibull"):
        parameters = {
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
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
            return -tf.reduce_sum(loglik_terms)
    
    elif(dist == "lognormal"):
        parameters = {
            "mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
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
            return -tf.reduce_sum(loglik_terms)
            
    elif(dist == "loglogistic"):
        parameters = {
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
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
            return -tf.reduce_sum(loglik_terms)
            
    elif( dist == "bs" ):        
        parameters = {
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
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
            return -tf.reduce_sum(loglik_terms)
    elif( dist == "bct" or dist == "box-cox-t" ):        
        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)

        parameters = {
            "mu": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "sigma": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "nu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "tau": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            mu = model.get_variable("mu")
            sigma = model.get_variable("sigma")
            nu = model.get_variable("nu")
            tau = model.get_variable("tau")

            target_shape = tf.shape(y)
            mu = tf.broadcast_to(mu, target_shape)
            sigma = tf.broadcast_to(sigma, target_shape)
            nu = tf.broadcast_to(nu, target_shape)
            tau = tf.broadcast_to(tau, target_shape)

            log_f_base = bct.log_pdf(y, mu, sigma, nu, tau)            
            log_S_base = bct.log_S(y, mu, sigma, nu, tau)            
            log_h_base = log_f_base - log_S_base
            
            loglik_terms = delta * log_h_base + log_S_base
            return -tf.reduce_sum(loglik_terms)

    elif( dist == "inv-gaussian" ):
        parameters = {
            "mu": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "lam": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
        }

        def loglikelihood_loss(model, nn_output, data):
            X, y, delta = data
            eps = tf.constant(1.0e-7, dtype = tf.float32) 
            
            mu = model.get_variable("mu")
            lam = model.get_variable("lam")

            y_safe = y + eps

            inv_gaussian_dist = tfp.distributions.InverseGaussian(loc = mu, concentration = lam)
            
            f0 = inv_gaussian_dist.prob(y_safe)
            f0 = tf.clip_by_value(f0, 1e-15, 1e15)
            log_f0 = tf.math.log(f0)
            
            S0 = inv_gaussian_dist.survival_function(y_safe)
            S0 = tf.clip_by_value(S0, 1e-15, 1.0)
            log_S0 = tf.math.log(S0)
            
            log_h0 = log_f0 - log_S0

            loglik_terms = delta * log_h0 + log_S0
            
            return -tf.reduce_sum(loglik_terms)

    elif(dist == "inv-weibull"):       
        parameters = {
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            k = model.get_variable("shape")
            lam = model.get_variable("scale")
        
            log_S0 = tf.math.log( 1 - tf.math.exp( -1/(y * lam)**k ) )
            log_f0 = tf.math.log(k) - k * tf.math.log(lam) - (k+1) * tf.math.log(y + eps) - 1.0/(y * lam)**k
            
            loglik_terms = delta * log_f0 + (1-delta) * log_S0
            return -tf.reduce_sum(loglik_terms)
            
    elif( dist == "gompertz" ):
        pass
    elif( dist == "inv-gompertz" ):
        parameters = {
            "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            X, y, delta = data
            
            eps = tf.constant(1.0e-7, dtype = tf.float32)
            y_safe = y + eps
            
            alpha = model.get_variable("shape")
            beta = model.get_variable("scale")
            
            alpha = tf.maximum(alpha, eps)
            beta = tf.maximum(beta, eps)

            beta_y = beta_y_raw = beta / y_safe
            e_beta_y_1 = tf.math.exp(beta_y) - 1.0
            alpha_beta = alpha / beta

            log_alpha = tf.math.log(alpha)
            log_y = tf.math.log(y_safe)
            log_f0 = log_alpha - 2.0 * log_y - alpha_beta * e_beta_y_1 + beta_y

            u = alpha_beta * e_beta_y_1
            S0 = 1.0 - tf.math.exp(-u)
            S0 = tf.clip_by_value(S0, 1e-15, 1.0)
            log_S0 = tf.math.log(S0)

            log_h0 = log_f0 - log_S0
            # log_f0 = tf.where(tf.math.is_finite(log_f0), log_f0, -250.0)
            # log_S0 = tf.where(tf.math.is_finite(log_S0), log_S0, -250.0)

            loglik_terms = delta * log_h0 + log_S0
            
            return -tf.reduce_sum(loglik_terms)
    
    elif( dist == "kwcwg" ):
        def logit(u):
            return -( tf.math.log(1-u) - tf.math.log(u) )
        
        parameters = {
            "a": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "b": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "alpha": {"link": tf.math.sigmoid, "link_inv": logit, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "gamma": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None},
            "lam": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None, "warmup_time": None}
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
            return -tf.reduce_sum(loglik_terms)
            
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

        if(warmup_times is None or par_name not in warmup_times):
            parameters[par_name]["warmup_time"] = 0
        else:
            parameters[par_name]["warmup_time"] = warmup_times[par_name]
    
    return parameters, loglikelihood_loss


def get_PH_model( X, dist, theta_init = None, beta_init = None, warmup_times = None ):
    '''
        Given a design matrix for the variables we have the interest of interpreting and the desired proportional hazards
        distribution, we create the corresponding thetaflow native function definitions of that model, considering the
        vector of linear coefficients to be the exact same size as X.

        X: Design matrix for interpretable data
        dist: Name of the distribution user wants to fit: ["exponential", "weibull", "lognormal", "loglogistic", "bs"]
        theta_init: Vector of initial values for base distribution parameters. It expects a dict whose keys
                    correspond to the specific distribution chosen.
                    List of parameter names:
                        - exponential: scale
                        - weibull: shape, scale
                        - lognormal: mu, scale
                        - loglogistic: shape, scale
                        - bs (Birnbaum-Saunders): shape, scale
    '''
    dist = dist.lower()
    if(dist == "exponential"):
        nn_output_size = 1

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        # parameters = {
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": beta_init, "warmup_time": 0}
        # }

        parameters = {
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }

        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)

            # Scale of the exponential distribution
            scale = model.get_variable("scale", nn_output)
            # Linear coefficients
            beta = model.get_variable("beta")[:,None]

            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z - tf.math.log( scale + eps ) ) - tf.math.exp(r_z) * y / scale
            return -tf.reduce_sum(loglik_terms)
            
    elif(dist == "weibull"):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        # parameters = {
        #     "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": beta_init, "warmup_time": 0}
        # }

        parameters = {
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            k = model.get_variable("shape", nn_output)
            lam = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]

            log_k = tf.math.log(k)
            log_lam = tf.math.log(lam)
            log_y = tf.math.log(y + eps)
            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_k - k * log_lam + (k-1)*log_y ) - tf.math.exp(r_z) * (y / lam)**k
            return -tf.reduce_sum(loglik_terms)
            
    elif(dist == "lognormal"):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)

        # parameters = {
        #     "mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": beta_init, "warmup_time": 0}
        # }
        
        parameters = {
            "mu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            mu = model.get_variable("mu", nn_output)
            scale = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]

            normal_dist = tfp.distributions.Normal(loc = mu, scale = scale)

            log_y = tf.math.log(y + eps)
            # Y ~ Lognormal(mu, sigma) => X = log(Y) ~ N(mu, sigma²)
            # P(Y > y) = P(log(Y) > log(y)) = P(X > log(y)) = S_N( log(y) )
            log_S0 = normal_dist.log_survival_function(log_y)
            # P(Y <= y) = Phi( (log(y) - mu) / sigma ) 
            # Therefore, f_Y(y) = phi( (log(y) - mu)/sigma ) * (y sigma)^(-1) => f_Y(y) = f_X( log_y ) / y
            log_f0 = normal_dist.log_prob(log_y) - log_y
            log_h0 = log_f0 - log_S0
            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) + tf.math.exp(r_z) * log_S0
            return -tf.reduce_sum(loglik_terms)
            
    elif(dist == "loglogistic"):
        nn_output_size = 2

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        # parameters = {
        #     "scale": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "shape": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1, "init": None, "warmup_time": 0},
        #     "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": beta_init, "warmup_time": 0}
        # }

        parameters = {
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("scale", nn_output)
            b = model.get_variable("shape", nn_output)
            beta = model.get_variable("beta")[:,None]

            log_y = tf.math.log(y + eps)
            log_b = tf.math.log(b)
            log_a = tf.math.log(a)
            log_terms = tf.math.softplus( b*( log_y - log_a ) )
            log_h0 = log_b - b * log_a + (b-1)*log_y - log_terms
            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) - tf.math.exp(r_z) * log_terms
            return -tf.reduce_sum(loglik_terms)
            
    elif( dist == "bs" ):
        nn_output_size = 2
        
        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        parameters = {
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            a = model.get_variable("shape", nn_output)
            b = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]

            pi = tf.constant( 3.141592653589793, dtype = tf.float32 )
            log_y = tf.math.log(y + eps)
            log_b = tf.math.log(b)
            log_a = tf.math.log(a)

            sqrt_b_y = tf.math.sqrt( b / (y+eps) )
            sqrt_y_b = tf.math.sqrt( y / b )
            log_f0 = -tf.math.log(2*tf.math.sqrt(2 * pi)) - log_a - log_b + tf.math.log( sqrt_b_y + tf.math.pow(sqrt_b_y, 3) ) - (y / b + b / (y + eps) - 2) / (2*a**2)
            z_score = (sqrt_y_b - sqrt_b_y) / a
            
            normal_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
            log_S0 = normal_dist.log_survival_function(z_score)
            
            log_h0 = log_f0 - log_S0
            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) + tf.math.exp(r_z) * log_S0
            return -tf.reduce_sum(loglik_terms)

    elif( dist == "bct" or dist == "box-cox-t" ):
        nn_output_size = 4

        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)

        parameters = {
            "mu": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "sigma": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "nu": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "tau": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            mu = model.get_variable("mu", nn_output)
            sigma = model.get_variable("sigma", nn_output)
            nu = model.get_variable("nu", nn_output)
            tau = model.get_variable("tau", nn_output)
            beta = model.get_variable("beta")[:,None]

            log_f0 = bct.log_pdf(y, mu, sigma, nu, tau)
            log_S0 = bct.log_S(y, mu, sigma, nu, tau)
            log_h0 = log_f0 - log_S0
            
            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) + tf.math.exp(r_z) * log_S0
            return -tf.reduce_sum(loglik_terms)
            
    elif( dist == "inv-gaussian" ):
        nn_output_size = 2
        
        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        parameters = {
            "mu": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "lam": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }

        def loglikelihood_loss(model, nn_output, data):
            X, z, y, delta = data
            eps = tf.constant(1.0e-7, dtype = tf.float32) 
            
            mu = model.get_variable("mu", nn_output)
            lam = model.get_variable("lam", nn_output)
            beta = model.get_variable("beta")[:,None]
            
            y_safe = y + eps

            inv_gaussian_dist = tfp.distributions.InverseGaussian(loc = mu, concentration = lam)
            
            f0 = inv_gaussian_dist.prob(y_safe)
            f0 = tf.clip_by_value(f0, 1e-15, 1e15)
            log_f0 = tf.math.log(f0)
            
            S0 = inv_gaussian_dist.survival_function(y_safe)
            S0 = tf.clip_by_value(S0, 1e-15, 1.0)
            log_S0 = tf.math.log(S0)
            
            log_h0 = log_f0 - log_S0

            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) + tf.math.exp(r_z) * log_S0
            
            return -tf.reduce_sum(loglik_terms)            
            
    elif(dist == "inv-weibull"):
        nn_output_size = 2
        
        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        parameters = {
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            # Unpack your data tuple
            X, z, y, delta = data
            eps = tf.constant(1.0e-07, dtype = tf.float32)
            
            k = model.get_variable("shape", nn_output)
            lam = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]
        
            log_f0 = tf.math.log(k) - k * tf.math.log(lam) - (k+1) * tf.math.log(y + eps) - 1.0/(y * lam)**k
            log_S0 = tf.math.log( 1 - tf.math.exp( -1/(y * lam)**k ) )
            log_h0 = log_f0 - log_S0
            
            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) + tf.math.exp(r_z) * log_S0
            
            return -tf.reduce_sum(loglik_terms)
            
    elif( dist == "inv-gompertz" ):
        nn_output_size = 2
        
        def softplus_inv(u):
            return tf.math.log(tf.math.exp(u) - 1)
        
        parameters = {
            "shape": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "scale": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": 1, "init": None, "warmup_time": None},
            "beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": X.shape[1], "init": None, "warmup_time": None}
        }
    
        def loglikelihood_loss(model, nn_output, data):
            X, z, y, delta = data
            
            eps = tf.constant(1.0e-7, dtype = tf.float32)
            y_safe = y + eps
            
            shape = model.get_variable("shape", nn_output)
            scale = model.get_variable("scale", nn_output)
            beta = model.get_variable("beta")[:,None]
            
            shape = tf.maximum(shape, eps)
            scale = tf.maximum(scale, eps)

            scale_y = scale / y_safe
            e_scale_y_1 = tf.math.exp(scale_y) - 1.0
            shape_scale = shape / scale

            log_shape = tf.math.log(shape)
            log_y = tf.math.log(y_safe)
            log_f0 = log_shape - 2.0 * log_y - shape_scale * e_scale_y_1 + scale_y

            u = shape_scale * e_scale_y_1
            S0 = 1.0 - tf.math.exp(-u)
            S0 = tf.clip_by_value(S0, 1e-15, 1.0)
            log_S0 = tf.math.log(S0)

            log_h0 = log_f0 - log_S0

            r_z = tf.matmul(z, beta)
            loglik_terms = delta * ( r_z + log_h0 ) + tf.math.exp(r_z) * log_S0
            
            return -tf.reduce_sum(loglik_terms)
        

    # Cycle through all parameters and set the initial value corresponding to a zero unconstrained value
    for par_name in parameters:
        if(par_name != "beta"):
            # If initial theta was not given, consider the initial value to be zero on the unconstrained scale
            if(theta_init is None or par_name not in theta_init):
                parameters[par_name]["init"] = parameters[par_name]["link"]( 0.0 )
            # Otherwise, just set the given values to each parameter
            else:
                parameters[par_name]["init"] = theta_init[par_name]

        if(warmup_times is None or par_name not in warmup_times):
            parameters[par_name]["warmup_time"] = 0
        else:
            parameters[par_name]["warmup_time"] = warmup_times[par_name]

    # If beta init was not given, simply set all coefficients to zeros
    if(beta_init is None):
        parameters["beta"]["init"] = np.repeat(0.0, X.shape[1])
    else:
        parameters["beta"]["init"] = beta_init

    return parameters, loglikelihood_loss, nn_output_size


def build_PH_model( Z, dist, theta_init = None, beta_init = None, warmup_times = None ):

    parameters, loglikelihood_loss, nn_output_size = get_PH_model( Z, dist = dist, theta_init = theta_init, beta_init = beta_init, warmup_times = warmup_times )

    def neural_network(model, seed = None):
        # initializer = tf.keras.initializers.GlorotNormal(seed = seed)
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

def get_survival_exp(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):

    if(simple):
        scale_train = model.predict("scale")
        scale_test = model.predict("scale")
    else:
        scale_train = model.predict(X_train)["scale"].numpy().flatten()
        scale_test = model.predict(X_test)["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

    S0_ts_train = np.exp( -ts_grid / scale_train )
    S0_train = np.exp( -y_train / scale_train )
    
    S0_ts_test = np.exp( -ts_grid / scale_test )
    S0_test = np.exp( - y_test / scale_test )

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )

    H_train = -np.log( S_train )
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

def get_survival_weibull(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):
    
    if(simple):
        k_train = model.predict("shape")
        lam_train = model.predict("scale")
        k_test = model.predict("shape")
        lam_test = model.predict("scale")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        k_train = pred_train["shape"].numpy().flatten()
        lam_train = pred_train["scale"].numpy().flatten()
        k_test = pred_test["shape"].numpy().flatten()
        lam_test = pred_test["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

    S0_ts_train = np.exp( -(ts_grid / lam_train)**k_train )
    S0_train = np.exp( -(y_train / lam_train)**k_train )
    
    S0_ts_test = np.exp( -(ts_grid / lam_test)**k_test )
    S0_test = np.exp( - (y_test / lam_test)**k_test )

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )
        
    H_train = -np.log( S_train )
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


def get_survival_lognormal(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):
    eps = tf.constant(1.0e-7)

    if(simple):
        mu_train = model.predict("mu")
        scale_train = model.predict("scale")
        mu_test = model.predict("mu")
        scale_test = model.predict("scale")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        mu_train = pred_train["mu"].numpy().flatten()
        scale_train = pred_train["scale"].numpy().flatten()
        mu_test = pred_test["mu"].numpy().flatten()
        scale_test = pred_test["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    log_ts_grid = tf.math.log(ts_grid + eps)
    
    normal_dist_train = tfp.distributions.Normal(loc = mu_train, scale = scale_train)

    S0_ts_train = normal_dist_train.survival_function(log_ts_grid).numpy()
    log_y_train = tf.math.log(y_train + eps)
    S0_train = normal_dist_train.survival_function(log_y_train).numpy()

    normal_dist_test = tfp.distributions.Normal(loc = mu_test, scale = scale_test)
    
    S0_ts_test = normal_dist_test.survival_function(log_ts_grid).numpy()
    log_y_test = tf.math.log(y_test + eps)
    S0_test = normal_dist_test.survival_function(log_y_test).numpy()

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )

    H_train = -np.log( S_train )
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


def get_survival_loglogistic(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):
    eps = tf.constant(1.0e-7, dtype = tf.float32)

    if(simple):
        b_train = model.predict("shape")
        a_train = model.predict("scale")
        b_test = model.predict("shape")
        a_test = model.predict("scale")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
    
        b_train = pred_train["shape"].numpy().flatten()
        a_train = pred_train["scale"].numpy().flatten()
        b_test = pred_test["shape"].numpy().flatten()
        a_test = pred_test["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    log_ts_grid = tf.math.log(ts_grid + eps)

    # -------------------- Train --------------------
    log_a_train = tf.math.log(a_train)
    log_terms_ts_train = tf.math.softplus( b_train*( log_ts_grid - log_a_train ) ).numpy()
    S0_ts_train = np.exp( -log_terms_ts_train )

    log_y_train = tf.math.log(y_train + eps)
    log_terms_y_train = tf.math.softplus( b_train*( log_y_train - log_a_train ) ).numpy()
    S0_train = np.exp( -log_terms_y_train )

    # -------------------- Test --------------------
    log_a_test = tf.math.log(a_test)
    log_terms_ts_test = tf.math.softplus( b_test*( log_ts_grid - log_a_test ) ).numpy()
    S0_ts_test = np.exp( -log_terms_ts_test )

    log_y_test = tf.math.log(y_test + eps)
    log_terms_y_test = tf.math.softplus( b_test*( log_y_test - log_a_test ) ).numpy()
    S0_test = np.exp( -log_terms_y_test )

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )

    H_train = -np.log( S_train )
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

def get_survival_bs(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):
    eps = tf.constant(1.0e-7, dtype = tf.float32)

    if(simple):
        a_train = model.predict("shape")
        b_train = model.predict("scale")
        a_test = model.predict("shape")
        b_test = model.predict("scale")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
    
        a_train = pred_train["shape"].numpy().flatten()
        b_train = pred_train["scale"].numpy().flatten()
        a_test = pred_test["shape"].numpy().flatten()
        b_test = pred_test["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    log_ts_grid = tf.math.log(ts_grid + eps)

    normal_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
    
    # -------------------- Train --------------------
    sqrt_b_ts_train = tf.math.sqrt( b_train / (ts_grid+eps) )
    sqrt_ts_b_train = tf.math.sqrt( ts_grid / b_train )
    z_ts_score_train = (sqrt_ts_b_train - sqrt_b_ts_train) / a_train
    S0_ts_train = normal_dist.survival_function(z_ts_score_train).numpy()

    sqrt_b_y_train = tf.math.sqrt( b_train / (y_train+eps) )
    sqrt_y_b_train = tf.math.sqrt( y_train / b_train )
    z_score_train = (sqrt_y_b_train - sqrt_b_y_train) / a_train
    S0_train = normal_dist.survival_function(z_score_train).numpy()

    # -------------------- Test --------------------
    sqrt_b_ts_test = tf.math.sqrt( b_test / (ts_grid+eps) )
    sqrt_ts_b_test = tf.math.sqrt( ts_grid / b_test )
    z_ts_score_test = (sqrt_ts_b_test - sqrt_b_ts_test) / a_test
    S0_ts_test = normal_dist.survival_function(z_ts_score_test).numpy()

    sqrt_b_y_test = tf.math.sqrt( b_test / (y_test+eps) )
    sqrt_y_b_test = tf.math.sqrt( y_test / b_test )
    z_score_test = (sqrt_y_b_test - sqrt_b_y_test) / a_test
    S0_test = normal_dist.survival_function(z_score_test).numpy()

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )

    H_train = -np.log( S_train )
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

def get_survival_bct(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):
    eps = tf.constant(1.0e-7, dtype = tf.float32)

    if(simple):
        mu_train = model.predict("mu")
        sigma_train = model.predict("sigma")
        nu_train = model.predict("nu")
        tau_train = model.predict("tau")
    
        mu_test = model.predict("mu")
        sigma_test = model.predict("sigma")
        nu_test = model.predict("nu")
        tau_test = model.predict("tau")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
    
        mu_train = pred_train["mu"].numpy().flatten()
        sigma_train = pred_train["sigma"].numpy().flatten()
        nu_train = pred_train["nu"].numpy().flatten()
        tau_train = pred_train["tau"].numpy().flatten()
    
        mu_test = pred_test["mu"].numpy().flatten()
        sigma_test = pred_test["sigma"].numpy().flatten()
        nu_test = pred_test["nu"].numpy().flatten()
        tau_test = pred_test["tau"].numpy().flatten()
        
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
    log_ts_grid = tf.math.log(ts_grid + eps)

    normal_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
    
    # -------------------- Train --------------------
    S0_ts_train = bct.S(ts_grid, mu_train, sigma_train, nu_train, tau_train).numpy()
    S0_train = bct.S(y_train, mu_train, sigma_train, nu_train, tau_train)

    # -------------------- Test --------------------
    S0_ts_test = bct.S(ts_grid, mu_test, sigma_test, nu_test, tau_test).numpy()
    S0_test = bct.S(y_test, mu_test, sigma_test, nu_test, tau_test)

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )

    H_train = -np.log( S_train )
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

def get_survival_inv_gaussian(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):

    if(simple):
        mu_train = model.predict("mu")
        lam_train = model.predict("lam")
        mu_test = model.predict("mu")
        lam_test = model.predict("lam")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        mu_train = pred_train["mu"].numpy().flatten()
        lam_train = pred_train["lam"].numpy().flatten()
        mu_test = pred_test["mu"].numpy().flatten()
        lam_test = pred_test["lam"].numpy().flatten()
        beta = model.predict("beta")[:,None]

    inv_gaussian_train = tfp.distributions.InverseGaussian(loc = mu_train, concentration = lam_train)
    inv_gaussian_test = tfp.distributions.InverseGaussian(loc = mu_test, concentration = lam_test)
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

    S0_ts_train = inv_gaussian_train.survival_function( ts_grid ).numpy()
    S0_train = inv_gaussian_train.survival_function( y_train ).numpy()
    S0_ts_test = inv_gaussian_test.survival_function( ts_grid ).numpy()
    S0_test = inv_gaussian_test.survival_function( y_test ).numpy()

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )
        
    H_train = -np.log( S_train )
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
            
    
def get_survival_inv_weibull(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):

    if(simple):
        k_train = model.predict("shape")
        lam_train = model.predict("scale")
        k_test = model.predict("shape")
        lam_test = model.predict("scale")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        k_train = pred_train["shape"].numpy().flatten()
        lam_train = pred_train["scale"].numpy().flatten()
        k_test = pred_test["shape"].numpy().flatten()
        lam_test = pred_test["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

    S0_ts_train = 1 - np.exp( -(1/(ts_grid * lam_train))**k_train )
    S0_train = 1 - np.exp( -(1/(y_train * lam_train))**k_train )
    
    S0_ts_test = 1 - np.exp( -(1/(ts_grid * lam_test))**k_test )
    S0_test = 1 - np.exp( -(1/(y_test * lam_test))**k_test )

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )
        
    H_train = -np.log( S_train )
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

def get_survival_inv_gompertz(model, y_train, z_train, X_train, y_test, z_test, X_test, ngrid = 100, simple = False):
    eps = tf.constant(1.0e-7)
    
    if(simple):
        shape_train = model.predict("shape")
        scale_train = model.predict("scale")
        shape_test = model.predict("shape")
        scale_test = model.predict("scale")
    else:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        shape_train = pred_train["shape"].numpy().flatten()
        scale_train = pred_train["scale"].numpy().flatten()
        shape_test = pred_test["shape"].numpy().flatten()
        scale_test = pred_test["scale"].numpy().flatten()
        beta = model.predict("beta")[:,None]
    
    ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

    scale_ts_train = scale_train / (ts_grid+eps)
    shape_scale_train = shape_train / scale_train
    e_scale_ts_1_train = tf.math.exp(scale_ts_train) - 1.0
    u_ts_train = shape_scale_train * e_scale_ts_1_train
    S0_ts_train = 1.0 - tf.math.exp(-u_ts_train)

    scale_y_train = scale_train / (y_train+eps)
    e_scale_y_1_train = tf.math.exp(scale_y_train) - 1.0
    u_y_train = shape_scale_train * e_scale_y_1_train
    S0_train = 1.0 - tf.math.exp(-u_y_train)
    
    scale_ts_test = scale_test / (ts_grid+eps)
    shape_scale_test = shape_test / scale_test
    e_scale_ts_1_test = tf.math.exp(scale_ts_test) - 1.0
    u_ts_test = shape_scale_test * e_scale_ts_1_test
    S0_ts_test = 1.0 - tf.math.exp(-u_ts_test)

    scale_y_test = scale_test / (y_test+eps)
    e_scale_y_1_test = tf.math.exp(scale_y_test) - 1.0
    u_y_test = shape_scale_test * e_scale_y_1_test
    S0_test = 1.0 - tf.math.exp(-u_y_test)

    if(simple):
        S_ts_train = S0_ts_train
        S_train = S0_train
        S_ts_test = S0_ts_test
        S_test = S0_test
    else:
        S_ts_train = S0_ts_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_train = S0_train**np.exp( np.dot( z_train, beta ).flatten() )
        S_ts_test = S0_ts_test**np.exp( np.dot( z_test, beta ).flatten() )
        S_test = S0_test**np.exp( np.dot( z_test, beta ).flatten() )
        
    H_train = -np.log( S_train )
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
    beta_summary
    return beta_summary