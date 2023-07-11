""" 
Changepoint analysis 
 
 Analyze keypoint trajectories to identify whether block structure exists in the data and what the temporal scale for the blocks are.

Methodology
 
 1. Compute a windowed derivative from each changepoint trajectory. 
 2. Use different window sizes: 100ms, 500ms, 1s, 2s and 5s to compute numerical derivatives
 3. z-score the derivative for each keypoint and binarize with a threshold
 4. Compute the number of threshold crossings on each frame and smooth with a Gaussian kernel to get the change score
 5. Find local peaks in the change score

Steps
 
 1. Read coordinates from sleap videos
 2. Flatten the data into frames x keypoint coordinates
 3. WAF a compute the filtered derivative for a given window for each 1D timeseries
 4. z-score and binarize each timeseries
 5. Count the number of threshold crossings for each frame - this is a 1D timeseries
 6. Smooth the changepoint timeseries with a Gaussian kernel

Generating the null-distribution to compute p-values
 
 1. For each experiment take the keypoint trajectories and cyclically permute them by a random interval
 2. Repeat the process of taking the derivative, z-score the derivative, binarize the z-scored derivative over a threshold and count the number of threshold crossings per frame across the different keypoint trajectories
 3. This will yield a distribution of change-scores for each frame.
 4. Compare the real change score for each frame with the change score of the null distribution and compute the p-value.
 5. The final change-score for the frame is -log(pval)

"""

import jax
print(jax.devices())
print(jax.__version__)

from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
# from scipy.stats import 

threshold = 1
sigma = 1
win_ms = 1000 
fps = 150
win_frames = int(fps*win_ms/1000)
repeats = 100


def derivate(keypoints, win=win_frames):
    """
    Parameters
    ----------
    keypoints : np.ndarray
        2D array of shape (n_frames, n_keypoints)
    win_frames : int
        Window size for the Savitzky-Golay filter
    
    """
    return savgol_filter(keypoints, window_length=win, axis=0, deriv=1, polyorder=3)

def zscore(arr):
    return (arr-arr.mean(axis=0))/arr.std(axis=0)

def binarize(arr, threshold):
    return np.abs(arr) > threshold

def count_crossings(arr):
    return np.sum(arr, axis=1)

def gauss_kern(ts, sig):
    return gaussian_filter1d(ts,sig)

def shuffle_ts(arr, interval=None):
    """
    Parameters
    ----------
    arr : np.ndarray
        2D array of shape (n_frames, n_keypoints)
    interval : int
        Interval to shift the array by
    """
    if interval is not None:
        return np.roll(arr, interval, axis=0)
    else:
        interval = np.random.randint(0, arr.shape[0])
        return np.roll(arr, interval, axis=0)


def real_distribution(arr):
        """
        This function takes a 2D array of shape (n_frames, n_keypoints)
        and computes the real distribution of
        changepoint scores for each frame.
        """

        return gauss_kern(
        count_crossings(
        binarize(
        zscore(
        derivate(
        shuffle_ts(arr)), threshold))), sigma)

    
def null_distribution(arr, repeats=repeats):
    """
    This function takes a 2D array of shape (n_frames, n_keypoints) 
    and computes the null distribution of 
    changepoint scores for each frame.
    """

    return gauss_kern(
        count_crossings(
        binarize(
        zscore(
        derivate(
        shuffle_ts(arr)), threshold))), sigma)
    

