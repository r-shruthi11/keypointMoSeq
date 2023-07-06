""" 
Write a function that takes a sleap tracked experiment directory and creates a config
compatible with keypoint_moseq. 

A complete config contains the following sections
        'video_dir': 'directory with videos from which keypoints were derived (used for crowd movies)',
        'bodyparts': 'used to access columns in the keypoint data',
        'skeleton': 'used for visualization only',
        'use_bodyparts': 'determines the subset of bodyparts to use for modeling and the order in which they are represented',
        'anterior_bodyparts': 'used to initialize heading',
        'posterior_bodyparts': 'used to initialize heading',
        'batch_length': 'data are broken up into batches to parallelize fitting',
        'trans_hypparams': 'transition hyperparameters',
        'ar_hypparams': 'autoregressive hyperparameters',
        'obs_hypparams': 'keypoint observation hyperparameters',
        'cen_hypparams': 'centroid movement hyperparameters',
        'error_estimator': 'parameters to convert neural net likelihoods to error size priors',
        'save_every_n_iters': 'frequency for saving model snapshots during fitting; if 0 only final state is saved', 
        'kappa_scan_target_duration': 'target median syllable duration (in frames) for choosing kappa',
        'whiten': 'whether to whiten principal components; used to initialize the latent pose trajectory `x`',
        'conf_threshold': 'used to define outliers for interpolation when the model is initialized',
        'conf_pseudocount': 'pseudocount used regularize neural network confidences',

These utils facilitate the creation of files and directories that can be an entrypoint to run sleap data
with keypoint_moseq.
The current design of this generates a config and data that can be sent to `format_data` in keypoint_moseq\project\io.py L231

format_data requires
    coordinates: a dict containing coordinates for a collection of sessions that are fit together where the values are numpy arrays of shape [frame, coordinates, 2] = [T, K, 2]

    confidences: None for now (my data are all very old versions of sleap tracked files that don't export confidences)
    
    bodyparts: a list of str with length K (same as in the coordinates array)

    batch_length: a scalar integer that maps the length of each batch 


"""

import sleap
import numpy as np
sleap.use_cpu_only()
import os
from pathlib import Path
import h5py
import tqdm
from textwrap import fill
from keypoint_moseq.project.io import generate_config
from scipy.interpolate import interp1d
import pandas as pd

def load_keypoints_from_slp_file(path_to_slp_file, use_instance=1, downsample=False, **kwargs):
    """
    This function takes a path to a single sleap-tracked experiment and looks for tracked and proofread .h5 files. 
    This is then used to load tracks of instance `use_instance` and returns the data in a numpy array with [frames, coordinates, 2] where the indices of coordinates to keep are specified in use_bodyparts.
    
    """
    def downsample_and_smooth(tracks, window_size=5, stride=3):
        """Smooth and downsample input data by averaging across a window."""
        T, K, _ = tracks.shape
        tracks_df = pd.DataFrame(tracks.reshape(T, -1), index=None)
        tracks_df_rolling_mean = tracks_df.rolling(window_size).mean()
        tracks_df_rolling_mean_stride = tracks_df.iloc[::stride, :].dropna()
        return tracks_df_rolling_mean_stride.to_numpy().reshape(-1, K, 2)

    def fill_missing_tracks(Y, kind="linear"):
        initial_shape = Y.shape
        # Flatten after first dim.
        Y = Y.reshape((initial_shape[0], -1))
        # Interpolate along each slice.
        for i in range(Y.shape[-1]):
            y = Y[:, i]
            non_missing_mask = ~np.isnan(y)
            num_non_missing = np.sum(non_missing_mask)
            # If we don't have enough points, don't interpolate, if we only have 2-3,
            # use linear interpolation, otherwise, use the interpolation requested.
            if num_non_missing <= 1:
                continue
            elif num_non_missing <= 4:
                kind_i = "linear"
            else:
                kind_i = kind
            # Build interpolant.
            x = np.flatnonzero(non_missing_mask)
            f = interp1d(x, y[x], kind=kind_i, fill_value=np.nan, bounds_error=False)
            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            Y[xq, i] = f(xq)
            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(Y[:, i])
            Y[:, i][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[:, i][~mask])
        #Restore to initial shape.
        Y = Y.reshape(initial_shape)
        return Y

    slp_dir = os.path.dirname(path_to_slp_file)
    path_to_tracking_h5 = os.path.join(slp_dir, 'inference.cleaned.proofread.tracking.h5')

    with h5py.File(path_to_tracking_h5, 'r') as f:
        tracks = np.transpose(np.copy(f['tracks'][use_instance, :, :, :]))
    
    # Crop to valid range
    last_fidx = np.argwhere(np.isfinite(tracks.reshape(len(tracks), -1)).any(axis=-1)).squeeze()[-1]
    tracks = tracks[:last_fidx]

    # Fill NaNs
    tracks = fill_missing_tracks(tracks)

    # Downsample
    if downsample:
        tracks = downsample_and_smooth(tracks)
    return tracks

def load_keypoints_from_slp_list(list_of_slp_expts, use_instance=1, downsample=False, **kwargs):
    """
    This function takes a list of sleap tracked experiments files. Specifically this contains a 
    list of strings pointing to the absolute path to */inference.proofread.cleaned.slp files which have been tracked and proofread. 
    It creates a dictionary `coordinates` which contains keys that correspond to a unique experiment ID and the values are numpy arrays of [T (frames), K (joints), 2 (xy)]
    """
    coordinates = {}
    print(use_instance)
    for expt_path in tqdm.tqdm(list_of_slp_expts, desc='Reading coordinates from sleap'):
        expt_id = os.path.split(os.path.dirname(expt_path))[-1]
        coordinates[expt_id] = load_keypoints_from_slp_file(expt_path, use_instance, downsample=downsample, **kwargs)
    return coordinates


def sleap_to_config(path_to_slp_file):
    """ 
    This function takes a path to a sleap-tracked experiment and generates a dlc-like 
    config that can be used to initialize a keypoint_moseq project and read in data. 
    A DLC config for keypoint_moseq requires the following (see: keypoint_moseq\project\io.py L205)
        - bodyparts
        - skeleton
        - use_bodyparts
        - video_dir
    From .slp skeleton
        i. generate 'bodyparts' list of bodyparts tracked
        ii. generate 'skeleton'
        iii. specify use_bodyparts
        iv. leave video_dir blank - this is irrelevant to my use case because I explicitly pass a list of tracked experiments to read in coordinates from
     
    """
    slp_config_dict = {}
    sleap_expt = sleap.load_file(path_to_slp_file)
    sleap_skel = sleap_expt.skeleton
    slp_config_dict['bodyparts'] = sleap_skel.node_names
    slp_config_dict['skeleton'] = [list(edge) for edge in sleap_skel.edge_names]
    slp_config_dict['use_bodyparts'] = sleap_skel.node_names # If needed this can be updated outside of this function
    slp_config_dict['video_dir'] = os.path.dirname(sleap_expt.video.filename)
    # slp_config_dict['anterior_bodyparts'] = ['thorax']
    # slp_config_dict['posterior_bodyparts'] = ['abdomen']
    slp_config_dict['use_instance'] = 1 # This is an integer index corresponding to the instance of interest
    return slp_config_dict 


def get_sleap_paths(video_dir):
    """Recursively searches subdirectories of video_dir to find **/*.inference.h5 files"""
    sleap_files = []
    for path in Path(video_dir).rglob('inference.h5'):
        print(path)
        sleap_files.append(str(path))
    return sleap_files


def setup_project_from_slp(project_dir, sample_slp_file=None, 
                  overwrite=False, **options):
    """
    Setup a project directory with the following structure
    ```
        project_dir
        └── config.yml
    ```
    
    Parameters
    ----------
    project_dir: str 
        Path to the project directory (relative or absolute)
        
    sample_slp_file: str, default=None
        Path to a deeplabcut config file. Relevant settings will be
        imported and used to initialize the keypoint MoSeq config.
        (overrided by **kwargs)
        
    overwrite: bool, default=False
        Overwrite any config.yml that already exists at the path
        ``[project_dir]/config.yml``
        
    **options
        Used to initialize config file
    """

    if os.path.exists(project_dir) and not overwrite:
        print(fill(f'The directory `{project_dir}` already exists. Use `overwrite=True` or pick a different name for the project directory'))
        return
        
    if sample_slp_file is not None: 
        slp_options = sleap_to_config(sample_slp_file)
                
        options = {**slp_options, **options}

    os.makedirs(project_dir, exist_ok=True)
    generate_config(project_dir, **options)


def find_matching_sleap_videos(keys, video_dir, as_dict=False):
    """
    This function assumes that all videos match this pattern
    video_dir/key/000000.mp4 where the keys are the expt_id.
    """
    video_paths = []
    for key in sorted(keys):
        matches = list(Path(video_dir).rglob(key))
        assert len(matches)==1, fill(f'No matches. Check paths and try again.')
        path_to_vid = os.path.join(matches[0], "000000.mp4")
        print(path_to_vid)
        is_path = Path(path_to_vid).is_file()
        assert is_path is True, fill(f'No matching videos found for {key}')
        video_paths.append(path_to_vid)
    if as_dict: return dict(zip(sorted(keys),video_paths))
    else: return video_paths

