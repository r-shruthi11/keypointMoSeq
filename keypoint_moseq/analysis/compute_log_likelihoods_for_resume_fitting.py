"""
Fitting sessions to data that don't all fit in memory requires
reading in batches of data that fitting continues on. This 
creates a series of checkpoints for each batch. 

To compute data log likelihood across batches, we need to iterate over 
the checkpoints and each batch of data and compute log P ( data | model params)
and normalize this by any baseline model - in this case the multivariate normal
"""

import jax
print(jax.devices())
print(jax.__version__)

from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm

import os
import matplotlib.pyplot as plt
import numpy as np
import glob

from jax_moseq.models.keypoint_slds import model_likelihood
import jax, os, joblib, tqdm, matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

def fit_mvn(data):
    """
    Multivariate gaussian model for the pose prediction to get the lower bound
    """

    n_features = data['Y'].shape[-2]
    d = data['Y'].shape[-1]

    x = data['Y'][data['mask'] > 0]
    x = x.reshape((-1, n_features * d))

    p = multivariate_normal(mean=np.mean(x, axis=0), cov=np.cov(x.T)).pdf(x)
    p = np.maximum(p, 1e-100)
    log_Y_given_mvn = np.sum(np.log(p))
    return log_Y_given_mvn


def find_checkpoint_paths(project_dir):
    
    checkpoint_paths = []
    
    checkpoint_paths = glob.glob(os.path.join(project_dir, '**', 'checkpoint.p'), recursive=True)
    
    for path in checkpoint_paths:
        print(os.path.dirname(path))

    return checkpoint_paths


def create_stats_folder(project_dir):
    
    stats_folder_path = os.path.join(project_dir, 'model_stats')
    os.makedirs(stats_folder_path, exist_ok=True)
    return stats_folder_path




def iterate_and_compute_likelihoods(project_dir):
    
    checkpoint_paths = find_checkpoint_paths(project_dir)
    
    all_log_normalized = {}

    for checkpoint_path in checkpoint_paths:
        session_id = os.path.split(os.path.dirname(checkpoint_path))[1]
        checkpoint = kpm.load_checkpoint(path=checkpoint_path)
        hypparams = jax.device_put(checkpoint['hypparams'])
        noise_prior = jax.device_put(checkpoint['noise_prior'])
        data = jax.device_put({'Y':checkpoint['Y'], 'mask':checkpoint['mask']})
        saved_iters = sorted(checkpoint['history'].keys())
        
        baseline_logY_mvn = fit_mvn(data)

        log_Y_given_model = []
        log_normalized = []

        for ix in tqdm.tqdm(saved_iters):
            states = jax.device_put(checkpoint['history'][ix]['states'])
            params = jax.device_put(checkpoint['history'][ix]['params'])
            ll = model_likelihood(data, states, params, hypparams, noise_prior)
            log_Y_given_model.append(ll['Y'].item())
            log_normalized.append(ll['Y'].item()/baseline_logY_mvn)
            # print(log_normalized)
            
        # all_log_normalized[session_id]['log_normalized'] = log_normalized
        # all_log_normalized[session_id]['checkpoint_path'] = checkpoint_path
        all_log_normalized[session_id] = log_normalized

    
    stats_folder = create_stats_folder(project_dir)
    save_llhs_path = os.path.join(stats_folder, 'all_log_normalized.p')

    joblib.dump(all_log_normalized, save_llhs_path)



if __name__ == "__main__":
    
    project_dir = "/scratch/gpfs/shruthi/pair_wt_gold/fitting/2023_04_26-21_11_07/"
    print("Running")
    iterate_and_compute_likelihoods(project_dir)

