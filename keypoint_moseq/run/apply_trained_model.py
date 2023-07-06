"""
Apply an existing kpms model to data that it was trained on to generate a results.h5 file.
1. Given an existing project directory and model folder name 
    a. reload the model checkpoint from this.
    b. reload the existing config from this folder
2. Given sleap_paths (used to fit the model) load coordinates, confidences (if any) and format the
input data.
3. Using this checkpoint, apply the model to the data. 
"""

# Imports
import os
import argparse
from rich.pretty import pprint

import keypoint_moseq as kpm
from keypoint_moseq.project.fit_utils import find_sleap_paths, load_coords_from_expts, find_model_name_in_project_dir

# Housekeeping
import jax
from jax.config import config
config.update('jax_enable_x64', False)
print(jax.devices())
print(jax.__version__)


def create_cli_parser():
    """Create a command line interface parser."""
    parser = argparse.ArgumentParser(description='Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.')
    
    parser.add_argument('--video_dir', type=str, default=r"D:\data\pair_wt_gold",
                        help='Path to directory containing sleap-tracked data.')
    
    parser.add_argument('--project_dir', type=str, default=r"D:\data\pair_wt_gold\fitting",
                        help='Path to directory where model will be saved.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name of model folder within project_dir.')
    parser.add_argument('--use_instance', type=int, default=1)
    
    return parser


def apply_kpms_model(coordinates, project_dir, name):
    """Load checkpoint, read config, apply model."""
    checkpoint = kpm.load_checkpoint(project_dir, name)
    config = kpm.load_config(project_dir)
    confidences = None
    kpm.apply_model(coordinates=coordinates, confidences=confidences,
                    project_dir=project_dir, **checkpoint, **config,
                    plot_every_n_iters=0, use_saved_states=False,
                    pca=kpm.load_pca(project_dir))


if __name__ == "__main__":
    # Parse arguments
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:")
    pprint(vars(args))
    print()

    # Read arguments
    video_dir = args.video_dir
    project_dir = args.project_dir
    name = args.model_name
    use_instance = args.use_instance

    # Setup project
    sleap_paths = find_sleap_paths(video_dir)

    # Read data
    coordinates = load_coords_from_expts(sleap_paths,
                                         project_dir,
                                         use_instance)
    if name is None:
        name = find_model_name_in_project_dir(project_dir)

    # Apply model
    apply_kpms_model(coordinates, project_dir, name)
