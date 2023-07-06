import operator
from functools import partial, reduce
import os
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import keypoint_moseq as kpm
from keypoint_moseq.project.fit_utils import find_model_name_in_project_dir


CONFIG_MAP_LIST = {"kappa": ["trans_hypparams", "kappa"],
                  "nlags": ["ar_hypparams", "nlags"],
                  "latent_dimension": ["latent_dimension"],
                  "alpha": ["trans_hypparams", "alpha"],
                  } 


test_data_folder = "/scratch/gpfs/shruthi/pair_wt_gold/190612_110405_wt_16276625_rig2.1/"
test_data_file = kpm.project.get_sleap_paths(test_data_folder)
coordinates = kpm.load_keypoints_from_slp_list(test_data_file)

def getFromDict(dataDict, mapList): # Usage: getFromDict(config, CONFIG_MAP_LIST['kappa'])
    return reduce(operator.getitem, mapList, dataDict)



def read_from_config(row):
    config = kpm.load_config(row)
    return getFromDict(config, CONFIG_MAP_LIST[HYPPARAM])



def get_model_name(row):
    name = find_model_name_in_project_dir(row)
    if name is not None:
        return os.path.join(row, name)
    else:
        return None   
    


def read_model_llh(row):
    name = find_model_name_in_project_dir(row)
    llh_file = os.path.join(row, name, 'llh.p')
    try:
        llh = joblib.load(llh_file)
    except FileNotFoundError:
        return None
    return llh['Y']



def get_syllables(results_path):
    try:
        results = kpm.load_hdf5(results_path)
    except FileNotFoundError:
        return None
    key = list(results.keys())[0]
    print(key)
    return np.copy(results[key]['syllables'])


    
def get_durations(z):
    stateseq_flat = z
    changepoints = np.insert(np.diff(stateseq_flat).nonzero()[0]+1,0,0)
    return changepoints[1:]-changepoints[:-1]    



def make_duration_plots(syll_arr):
        durations = get_durations(syll_arr)
        usages = kpm.get_usages(syll_arr)
        min_usage = 0.005

        fig, axs = plt.subplots(1, 2, figsize=(18, 4))
        plt.suptitle(f'HYPPARAM = {HYPPARAM}')
        
        usages = np.sort(usages[usages>min_usage])[::-1]
        axs[0].bar(range(len(usages)),usages,width=1)
        axs[0].set_ylabel('probability')
        axs[0].set_xlabel('syllable rank')
        axs[0].set_title('Usage distribution')

        lim = int(np.percentile(durations, 95))
        binsize = max(int(np.floor(lim/30)),1)
        axs[1].hist(durations, range=(1,lim), bins=(int(lim/binsize)), density=True)
        axs[1].set_xlim([1,lim])
        axs[1].set_xlabel('syllable duration (frames)')
        axs[1].set_ylabel('probability')
        axs[1].set_title(f'Duration distribution')
        axs[1].set_xticks(np.arange(0, lim, 10))

        med_durs = np.median(durations)
        axs[1].axvline(med_durs, color='k')
    



def compute_median_syllable_duration(row):
    results_path = row['results_path']
    syllables = get_syllables(results_path)
    if syllables is not None:
        durations = get_durations(syllables)
        return np.median(durations)
    else:
        return None



def create_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, required=True, help='Path to the array_args_file')
    parser.add_argument('--hyp_param', type=str, required=True, help='Hyperparameter info to analyze')
    return parser



def read_paths_to_models(args_path):
    with open(args_path, 'r') as txt_file:
        paths = txt_file.readlines()
    paths = [str.replace(path, '\n', '') for path in paths if HYPPARAM in path]
    hyp_df = pd.DataFrame(paths, index=None, columns=["Path"])
    print(hyp_df.head())
    return hyp_df


def save_hyp_df(hyp_df, args_path):
    save_path = os.path.join(os.path.dirname(args_path), f'hypparam{HYPPARAM}_sweep_stats.csv')
    hyp_df.to_csv(save_path, index=False)
    print(f'Saved hyperparameter sweep stats to {save_path}')



def make_movies(row):
    project_dir = row['Path']
    results_path = row['results_path']
    name = find_model_name_in_project_dir(project_dir)

    kpm.update_config(project_dir, video_dir='/tigress/MMURTHY/junyu/data/pair')
    config = kpm.load_config(project_dir)
    kpm.generate_crowd_movies(project_dir=project_dir, results_path=results_path, **config, 
                                sleap=True, name=name)



def generate_trajectory_plots(row):
    config = kpm.load_config(row['Path'])
    kpm.generate_trajectory_plots(coordinates, results_path=row['results_path'],
                                  **config, output_dir=os.path.dirname(row['results_path']))


def main():
    parser = create_cli_parser()
    args = parser.parse_args()
    args_path = args.args_path
    global HYPPARAM
    HYPPARAM = args.hyp_param
    
    hyp_df = read_paths_to_models(args_path)
    hyp_df['val_hypparam'] = hyp_df['Path'].apply(read_from_config)
    hyp_df['model_path'] = hyp_df['Path'].apply(get_model_name)
    hyp_df['llh'] = hyp_df['Path'].apply(read_model_llh)
    hyp_df['results_path'] = hyp_df['model_path'].apply(lambda row: os.path.join(row, 'results.h5'))
    hyp_df['median_duration'] = hyp_df.apply(compute_median_syllable_duration, axis=1)


    hyp_df.apply(make_movies, axis=1)
    hyp_df.apply(generate_trajectory_plots, axis=1)
    save_hyp_df(hyp_df, args_path)
    



if __name__ == '__main__':
    main()

