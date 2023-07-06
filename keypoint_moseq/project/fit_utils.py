"""

"""
import os
import numpy as np
from textwrap import fill
from datetime import datetime
import tqdm
import keypoint_moseq as kpm
from keypoint_moseq.project.fitting import resample_model, update_history, fit_model
from keypoint_moseq.project.viz import plot_progress
from keypoint_moseq.project.io import save_checkpoint
from keypoint_moseq.run.constants import EXPT_BATCH_LEN
from jax_moseq.models.keypoint_slds import model_likelihood


def find_sleap_paths(video_dir):
    """Search recursively within video_dir to find paths to sleap-tracked expts. and files."""

    print(f"Searching for paths within {video_dir}")
    sleap_paths = kpm.project.get_sleap_paths(video_dir)
    print(f"Found {len(sleap_paths)} expts.")
    return sleap_paths


def find_model_name_in_project_dir(project_dir):
    """Find the model_name in the project_dir.
    """
    # Find model_name in project_dir
    model_name = None
    for name in os.listdir(project_dir):
        if os.path.isdir(os.path.join(project_dir, name)):
            if os.path.exists(os.path.join(project_dir, name, "checkpoint.p")):
                model_name = name
                break
    if model_name is None:
        # raise ValueError("No models found in project_dir.")
        print(f"No models found in {project_dir}.")
    return model_name


def load_coords_from_expts(sleap_paths, project_dir, use_instance):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    """
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths, use_instance)
    print("Printing summary of data loaded.")
    for k,v in coordinates.items():
        print(f"Session {k}")
        print(f"Data: {v.shape}")
    return coordinates


def load_data_from_expts(sleap_paths, project_dir, use_instance):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    Format data for modelling and move to GPU.
    """
    config = kpm.load_config(project_dir)
    coordinates = load_coords_from_expts(sleap_paths, project_dir, use_instance)
    data, batch_info = kpm.format_data(coordinates, **config)
    return data, batch_info


def run_fit_PCA(data, project_dir, cv_split_save_dir):
    """
    Fit PCA to data using parameters defined in config which is read from project_dir.
    Save PCA to project_dir.
    Run scripts to plot and visualize results of PCA fitting. (TODO: How does this work when run from the CL?)
    """
    config = kpm.load_config(project_dir)
    pca = kpm.fit_pca(**data, **config, conf=None)
    kpm.save_pca(pca, cv_split_save_dir)
    kpm.print_dims_to_explain_variance(pca, 0.9)
    # Visualization might cause CL calls to crash
    # so comment and run these lines locally
    # kpm.plot_scree(pca, project_dir=project_dir)
    # kpm.plot_pcs(pca, project_dir=project_dir, **config)


def fit_keypoint_ARHMM(project_dir, cv_split_save_dir, data, batch_info, name=None, return_checkpoint_path=False):
    """
    Load PCA from project_dir and initialize the model.
    Fit AR-HMM
    """
    config = kpm.load_config(project_dir)
    pca = kpm.load_pca(cv_split_save_dir)
    model = kpm.initialize_model(pca=pca, **data, **config)
    # TODO: WAF to visualize initialized parameters

    model, history, name = fit_model(model, data, batch_info, batch=-1, ar_only=True,
                                   num_iters=50, save_dir=cv_split_save_dir,
                                   plot_every_n_iters=0, name=name)
    # TODO: WAF to visualize AR-HMM fit parameters
    return model, history, name


# def fit_keypoint_SLDS(project_dir, name):
#     """
#     Restore checkpoint of model in project_dir/name.
#     Resume fitting with SLDS.
#     """
#     checkpoint = kpm.load_checkpoint(project_dir=project_dir, name=name)
#     model, history, name = kpm.resume_fitting(**checkpoint, project_dir=project_dir,
#                                         ar_only=False, num_iters=200, conf=None,
#                                         plot_every_n_iters=0)


def resume_slds_fitting_to_new_data(sleap_paths, checkpoint_path,
                                    project_dir, save_dir):
    """
    Load checkpoint, initialize model and resume fitting to new data.

    Parameters
    ----------
    sleap_paths : list
        List of paths to sleap-tracked experimental sessions to fit model to.
    checkpoint_path : str
        Path to model checkpoint to resume fitting from.
    project_dir : str
        Path to directory to read properties and config for the fit.
    save_dir : str
        Path to directory where model will be saved.
    """

    config = kpm.load_config(project_dir)
    pca = kpm.load_pca(save_dir)
    use_instance = config["use_instance"]

    # Hack here to resume fitting when jobs crashed at i = 20
    # sleap_paths = sleap_paths[20:]
    # print(f"Hack here to resume fitting from batch starting at idx = 20. \n")
    # print(f"The paths: {sleap_paths} \n")

    # Split sleap_paths into batches of length expt_batch_length
    for b in range(0, len(sleap_paths), EXPT_BATCH_LEN):

        # Batch expt. paths
        sleap_paths_batch = sleap_paths[b:b + EXPT_BATCH_LEN]
        current_batch = b // EXPT_BATCH_LEN
        print(f"Fitting to batch {current_batch} of {len(sleap_paths) // EXPT_BATCH_LEN}")
        data, batch_info = load_data_from_expts(sleap_paths_batch,
                                                project_dir,
                                                use_instance)

        # Load checkpoint
        if b == 0:
            checkpoint = kpm.load_checkpoint(path=checkpoint_path)
        else:
            checkpoint = kpm.load_checkpoint(save_dir, name)

        # Initialize a new model using saved parameters
        model = kpm.initialize_model(pca=pca, **data,
                                     params=checkpoint["params"],
                                     **config)

        # Resume fitting with new data
        model, history, name = fit_model(model, data, batch_info, current_batch, ar_only=False,
                                             num_iters=100, project_dir=project_dir, save_dir=save_dir,
                                             plot_every_n_iters=0)

    return name


def calculate_ll(states, params, hypparams, noise_prior, data):
    # Compute log likelihoods
    ll = model_likelihood(data, states, params, hypparams, noise_prior)
    log_Y_and_model = np.sum([v.item() for v in ll.values()])
    log_Y_given_model = ll['Y'].item()
    return log_Y_and_model, log_Y_given_model
