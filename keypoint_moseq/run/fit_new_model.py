"""
Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.
"""

# Imports
import argparse
from rich.pretty import pprint
import jax
from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm
from pathlib import Path
from datetime import datetime
import os

from hyperparams.fit_hyperparams_sleap import train
from keypoint_moseq.project.fit_utils import find_sleap_paths

# Housekeeping
print(jax.devices())
print(jax.__version__)

print(os.environ)


def create_cli_parser():
    """Create a command line interface parser."""
    parser = argparse.ArgumentParser(description='Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.')
    
    parser.add_argument('--video_dir', type=str, default=r"D:\data\pair_wt_gold",
                        help='Path to directory containing sleap-tracked data.')
    
    parser.add_argument('--project_dir', type=str, default=r"D:\data\pair_wt_gold\fitting",
                        help='Path to directory where model will be saved.')
    
    parser.add_argument('--use_instance', type=int, default=1)
    
    return parser


def create_folder_for_project(project_dir, sample_slp_path, use_instance, hyper, use_bodyparts):
    """
    Given a base directory where results will be saved,
    create a folder for each hyperparameter sweep.

    Parameters
    ----------
    project_dir : str
        Path to base directory where results will be saved.
    sample_slp_path : str
        Path to sleap-tracked expt. to be used to initialize config.yml
    use_instance : int
        Instance of keypoints to be used for modelling. O for female, 1 for male
    hyper : dict
        Dictionary of hyperparameters to be used for this run.
    """

    # Create base folder to save results
    if project_dir is None:
        project_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/fitting/"
    base_date_string = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    save_results_to = os.path.join(project_dir, base_date_string)
    print(f"Saving project to: {save_results_to}")

    # Get expt. paths to be used to generate base config
    print(f"Sample sleap path: {sample_slp_path}")

    # Setup text file to log array args for slurm run
    Path(save_results_to).mkdir(parents=True, exist_ok=True)

    # Setup project directory with custom configs
    update_dict = {'use_instance': use_instance}
    for k, v in hyper.items():  # Iterate over hyperparam keys
        kpm.setup_project_from_slp(save_results_to,
                                   sample_slp_file=sample_slp_path,
                                   overwrite=True)  # Make sure a sample slp file exists
        if k == 'kappa':
            update_dict.update({k: float(v)})  # YAML writing doesn't support np.float64 or np.int64 datatypes
        if k == 'latent_dimension':
            update_dict.update({k: int(v)})
        if k == 'alpha':
            update_dict.update({k: float(v)})
        if k == 'nlags':
            update_dict.update({k: int(v)})

    print("Updating config with hyperparameters: ")
    pprint(update_dict)
    kpm.update_config(save_results_to,
                      use_bodyparts=use_bodyparts,
                      anterior_bodyparts=['thorax'],
                      posterior_bodyparts=['abdomen'],
                      slope=-0.47,
                      intercept=0.236721,
                      PCA_fitting_num_frames=270000,
                      keypoint_colormap='cool',
                      **update_dict)

    config = kpm.load_config(save_results_to)
    print("Config for this run: ")
    pprint(config)

    # Print base folder and array_args file name
    print(f"Saving results to: {save_results_to}")
    return save_results_to


if __name__ == "__main__":

    HYPER = {"nlags": 3,
             "kappa": 1e12,
             "alpha": 100,
             "latent_dimension": 16,
             "gamma": 1e3}

    use_bodyparts = ['thorax', 'abdomen', 'wingL',
                     'wingR', 'forelegL4', 'forelegR4',
                     'midlegL4', 'midlegR4', 'hindlegL4',
                     'hindlegR4']


    # Parse arguments
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:")
    pprint(vars(args))
    print()

    # Read arguments
    video_dir = args.video_dir
    project_dir = args.project_dir
    use_instance = args.use_instance

    hyper = HYPER
    print(f"Hyperparameters for this run: ")
    pprint(hyper)

    # Setup project
    sleap_paths = find_sleap_paths(video_dir)
    sample_sleap_path = sleap_paths[0]
    save_results_to = create_folder_for_project(project_dir,
                                                sample_sleap_path,
                                                use_instance,
                                                hyper,
                                                use_bodyparts)

    model_name = train(sleap_paths, save_results_to, save_results_to, use_instance)
