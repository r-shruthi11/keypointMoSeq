"""

"""
import jax
print(jax.devices())
print(jax.__version__)

import numpy as np
import matplotlib.pyplot as plt

from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm

from datetime import datetime
import os
from pathlib import Path

# Define hyperparam sweep configuration
hyper = {"kappa": list(np.logspace(2, 12, num=6, base=10,dtype=float)),
        "nlags": [1, 3, 5, 10],
        "alpha": [0.1, 0.5, 1, 5, 10, 50],
        "latent_dimension": list(np.arange(2, 22, 2, dtype=int))}

# Define base configuration
base = {"kappa": 1e6,
       "alpha": 5.7, 
       "nlags": 3,
       "latent_dimension": 10}

# Define use_bodyparts for sleap experiments
use_bodyparts = ['thorax', 'abdomen', 'wingL',
                 'wingR', 'forelegL4', 'forelegR4',
                 'midlegL4', 'midlegR4', 'hindlegL4',
                 'hindlegR4']


def print_hyper_config(hyper):
    """
    Given a dictionary of hyperparameters 
    and their values, print the keys, values
    and type.

    Parameters
    ----------
    hyper: dict,
        dictionary with keys that map onto specific hyperparam names
        and values (or list of values) that map onto specific 
        hyperparam values
    """
    for key in hyper.keys():
        print(f"{key}: {hyper[key]}")
        print(f"Datatype: {type(hyper[key])}")

def create_folders_for_hyperparam(video_dir, hyper, save_data_to=None):
    """
    Given a base directory where results will be saved, 
    create a folder for each hyperparameter sweep.

    Parameters
    ----------
    video_dir: str,
        directory where sleap experiments are stored

    hyper: dict,
        dictionary with keys that map onto specific hyperparam names

    save_data_to: str, 
        directory where results will be saved
    """
    # Create base folder to save results
    if save_data_to is None:
        save_data_to = r"/scratch/gpfs/shruthi/pair_wt_gold/fitting/"
    base_date_string = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    base_folder = os.path.join(save_data_to, base_date_string)
    print(f"Saving hyperparam sweep runs to: {base_folder}")

    # Get expt. paths to be used to generate base config 
    sleap_paths = kpm.project.find_sleap_paths(video_dir)
    sample_sleap_path = sleap_paths[0]
    print(f"Sample sleap path: {sample_sleap_path}")

    # Setup text file to log array args for slurm run
    txt_file_name = os.path.join(base_folder, 'array_args.txt')

    # Setup project directories with custom configs
    with open(txt_file_name, 'w') as txt_file:
        for k, v in hyper.items(): # Iterate over hyperparam keys
            save_folder = os.path.join(base_folder, f"sweep_{k}")
            for vi in range(len(v)): # Iterate over values for each hyperparam
                project_dir = os.path.join(save_folder, str(int(vi)))
                Path(project_dir).mkdir(parents=True, exist_ok=True)
                kpm.setup_project_from_slp(project_dir,
                                        sample_slp_file=sample_sleap_path,
                                        overwrite=True) # Make sure a sample slp file exists
                if k == 'kappa':
                    update_dict = {k: float(v[vi])} # YAML writing doesn't support np.float64 or np.int64 datatypes
                elif k == 'latent_dimension':
                    update_dict = {k: int(v[vi])}

                kpm.update_config(project_dir, 
                                use_bodyparts=use_bodyparts,
                                anterior_bodyparts=['thorax'],
                                posterior_bodyparts=['abdomen'],
                                latent_dimension=10,
                                slope= -0.47,
                                intercept= 0.236721,
                                pca_fitting_num_frames = 270000,
                                keypoint_colormap = 'cool',
                                **update_dict)
            
                txt_file.write(project_dir+"\n") # Write project directory to text file
    
    # Print base folder and array_args file name
    print(f"Base folder: {base_folder}")
    print(f"Array args file name: {txt_file_name}")
    
    return base_folder, txt_file_name

def setup_sleap_paths(video_dir):
    """Given a directory where sleap experiments are stored, 
    return a list of paths to each sleap experiment."""
    sleap_paths = kpm.project.find_sleap_paths(video_dir)
    return sleap_paths
