import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
import os
import shutil
import sys
import time
sys.path.append("../models/GPA")
# from useful_functions import *


def generate_random_simulate_image(mean, sigma):
    """Generate a random simulated image following a truncated normal distribution."""
    tfd = tfp.distributions
    dist = tfd.TruncatedNormal(loc=mean, scale=sigma, low=[0.], high=[1.])
    simulate_img = dist.sample(1)
    return simulate_img


# def compute_true_density(tick_list, mean, sigma, IsSave=False):
#     """
#     Compute the true (analytical) density of a truncated normal distribution.

#     Args:
#         tick_list : list of evaluation points.
#         mean, sigma : parameters of the truncated normal distribution.
#         IsSave : if True, save the computed density to ./f_true.pkl.
#     """
#     tfd = tfp.distributions
#     dist = tfd.TruncatedNormal(loc=mean, scale=sigma, low=[0.], high=[1.])
#     f_true = tf.concat([dist.prob(tick) for tick in tick_list], axis=0)
#     print("True density shape:", f_true.shape)
#     if IsSave:
#         with open("./f_true.pkl", 'wb') as f:
#             pickle.dump(f_true, f)
#             print("save true density at ./f_true.pkl")
#     return f_true


def generate_simulate_data(path, N, mean, sigma):
    """
    Generate N simulated images and save them as .npy files in the given path.

    Args:
        path  : output directory.
        N     : number of simulated images.
        mean, sigma : distribution parameters.
    """
    if os.path.exists(path):  # If the folder exists, delete it
        shutil.rmtree(path)   # Recursively delete the directory
    os.mkdir(path)            # Create a new empty directory

    # Generate new simulated images
    train_list = []
    for i in range(N):
        train_img = generate_random_simulate_image(mean, sigma)
        train_path = path + f"train_img_{i}.npy"
        np.save(train_path, train_img.numpy())
        train_list.append(train_path)
    return train_list


def compute_CD_matrix(path, N, G, p, q, bandwidth, tick_tensor):
    """
    Compute the classical (CD) nonparametric density estimator.

    Args:
        path        : path to simulated image .npy files.
        N           : number of simulated images.
        G, p, q     : grid / image dimensions.
        bandwidth   : kernel bandwidth.
        tick_tensor : evaluation tensor.

    Returns:
        CD_tensor   : estimated classical density tensor.
    """
    CD_tensor = tf.zeros((G, p, q), dtype=tf.float32)
    for i in range(N):
        train_img = tf.constant(np.load(path + f"/train_img_{i}.npy"))
        # compute classic nonparametric density estimator
        tmp_tensor = 1 / tf.sqrt(2 * np.pi) * tf.exp(-(train_img - tick_tensor) ** 2 / (2 * bandwidth ** 2))
        tmp_tensor = tmp_tensor / (N * bandwidth)
        CD_tensor += tmp_tensor
    return tf.squeeze(CD_tensor)


def compute_DS_matrix(CD_est, location_weight):
    """
    Compute the doubly smoothed (DS) estimator by spatial convolution.

    Args:
        CD_est         : classical density estimation tensor.
        location_weight: spatial weighting (smoothing) kernel.

    Returns:
        DS_est         : doubly smoothed density estimate.
    """
    CD_est = tf.squeeze(CD_est)
    CD_est = tf.reshape(CD_est, [1, *CD_est.shape, 1])
    Omega1 = tf.nn.depthwise_conv2d(CD_est, location_weight, strides=[1, 1, 1, 1], padding='SAME')
    Omega2 = tf.reduce_sum(location_weight)
    DS_est = Omega1 / Omega2
    return DS_est
