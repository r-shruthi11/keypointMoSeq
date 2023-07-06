"""

"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from keypoint_moseq.project.io import load_checkpoint, save_llh
from keypoint_moseq.project.fit_utils import calculate_ll


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


def get_logll_from_checkpoint(ckp):
    log_Y_and_model = []
    log_Y_given_model = []

    data = {'Y': ckp['Y'], 'mask': ckp['mask']}
    n_samples = np.sum(data['mask']).astype(int)
    log_Y_given_mvn = fit_mvn(data)
    for iter in ckp['history']:
        states = ckp['history'][iter]['states']
        params = ckp['history'][iter]['params']
        hypparams = ckp['hypparams']
        noise_prior = ckp['noise_prior']
        l1, l2 = calculate_ll(states, params, hypparams, noise_prior, data)
        log_Y_and_model.append(l1)
        log_Y_given_model.append(l2)
    return log_Y_and_model, log_Y_given_model, n_samples, log_Y_given_mvn


def process_checkpoints(project_dir):
    llh_df = pd.DataFrame(columns=['hyp', 'hyp_idx', 'cv_idx', 'batch', 'n_iters', 'n_samples',
                                   'log_Y_given_mvn', 'log_Y_and_model', 'log_Y_given_model', 'bits'])
    sweep_dirs = [d for d in os.listdir(project_dir) if 'sweep_' in d]
    for hyp_sweep_dir in sweep_dirs:
        hyp = hyp_sweep_dir.split('_')[-1]
        print("sweep_hyp", hyp)
        for hyp_val_dir in glob.glob(os.path.join(project_dir, hyp_sweep_dir) + '/*'):
            hyp_idx = hyp_val_dir.split('/')[-1]
            print("each_sweep_dir", hyp_idx)
            for cv_split_path in glob.glob(hyp_val_dir + '/*'):
                if not os.path.isdir(cv_split_path):
                    continue
                cv_idx = cv_split_path.split('/')[-1]

                n_iters = []
                n_samples = []
                log_Y_given_mvn = []
                log_Y_and_model = []
                log_Y_given_model = []
                bits = []

                for batch_ckp_path in glob.glob(cv_split_path + '/*'):
                    if not os.path.isdir(batch_ckp_path):
                        continue
                    print("loading...", batch_ckp_path)

                    ckp = load_checkpoint(path=batch_ckp_path + '/checkpoint.p')
                    b = ckp['current_batch']
                    b_log_Y_and_model, b_log_Y_given_model, b_n_samples, b_log_Y_given_mvn = get_logll_from_checkpoint(ckp)
                    b_n_iters = list(ckp['history'].keys())
                    b_bits = ((np.array(b_log_Y_given_model) - b_log_Y_given_mvn) / b_n_samples).tolist()

                    log_Y_and_model.append(b_log_Y_and_model)
                    log_Y_given_model.append(b_log_Y_given_model)
                    bits.append(b_bits)
                    n_samples.append(b_n_samples)
                    log_Y_given_mvn.append(b_log_Y_given_mvn)
                    n_iters.append(b_n_iters)

                    llh_df.loc[len(llh_df)] = [hyp, hyp_idx, cv_idx, b, b_n_iters, b_n_samples, b_log_Y_given_mvn,
                                               b_log_Y_and_model, b_log_Y_given_model, b_bits]

    return llh_df


if __name__ == '__main__':

    project_dir = '/scratch/gpfs/us3519/fit_pair/project/2023_07_05-02_33_31/'
    llh_df = process_checkpoints(project_dir)
    # save_llh(llh_df, project_dir)

    # Plot data LL, and bits across CV train splits
    run_nlags_0_df = llh_df[llh_df['hyp'] == 'nlags'][llh_df['hyp_idx'] == '0'].sort_values(['cv_idx', 'batch']).groupby(['cv_idx'])

    # # plot log_Y_given_model, non-normalized so probably doesn't make sense
    # data_ll = np.array(run_nlags_0_df.agg({'log_Y_given_model': 'sum'})['log_Y_given_model'].tolist()).Tq
    # print(data_ll)
    # plt.plot(data_ll, linewidth=1)
    # plt.errorbar(np.arange(len(data_ll)), np.mean(data_ll, axis=1), yerr=np.std(data_ll, axis=1), fmt='-o')
    # plt.xlabel('iterations')
    # plt.ylabel('log(Y|slds)')
    # plt.title('log(Y|slds) for CV splits')
    #
    # # plot bits/frame normalized w.r.t mvn
    # bits_ = np.array(run_nlags_0_df.agg({'bits':'sum'})['bits'].tolist()).T
    # print(bits_)
    # plt.plot(bits_)
    # plt.xlabel('iterations')
    # plt.ylabel('information w.r.t. mvn')
    # plt.title('bits/frame for CV splits')
    #
    # plt.show()


