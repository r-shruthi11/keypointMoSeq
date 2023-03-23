import joblib
import os
import numpy as np
import tqdm
import jax
import warnings
warnings.formatwarning = lambda msg, *a: str(msg)
from textwrap import fill
from datetime import datetime
from keypoint_moseq.model.gibbs import resample_model
from keypoint_moseq.model.initialize import initialize_model
from keypoint_moseq.project.viz import plot_progress
from keypoint_moseq.util import get_durations, batch, unbatch, estimate_coordinates, get_usages
from keypoint_moseq.project.io import save_checkpoint, format_data, save_hdf5
    

def update_history(history, iteration, model, include_states=True): 
    
    model_snapshot = {
        'params': jax.device_get(model['params']),
        'seed': jax.device_get(model['seed'])}
    
    if include_states: 
        model_snapshot['states'] = jax.device_get(model['states'])
        
    history[iteration] = model_snapshot
    return history



def fit_model(model,
              data,
              batch_info,
              start_iter=0,
              history=None,
              verbose=True,
              num_iters=50,
              ar_only=False,
              name=None,
              project_dir=None,
              save_data=True,
              save_states=True,
              save_history=True,
              save_every_n_iters=10,
              history_every_n_iters=10,
              states_in_history=True,
              plot_every_n_iters=10,  
              save_progress_figs=True,
              **kwargs):
    
    
    if save_every_n_iters>0 or save_progress_figs:
        assert project_dir, fill(
            'To save checkpoints or progress plots during fitting, provide '
            'a ``project_dir``. Otherwise set ``save_every_n_iters=0`` and '
            '``save_progress_figs=False``')
        if name is None: 
            name = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        savedir = os.path.join(project_dir,name)
        if not os.path.exists(savedir): os.makedirs(savedir)
        print(fill(f'Outputs will be saved to {savedir}'))

    
    if history is None: history = {}

    for iteration in tqdm.trange(start_iter, num_iters+1):
        if history_every_n_iters>0 and (iteration%history_every_n_iters)==0:
            history = update_history(history, iteration, model, 
                                     include_states=states_in_history)
            
        if plot_every_n_iters>0 and (iteration%plot_every_n_iters)==0:
            plot_progress(model, data, history, iteration, name=name, 
                          savefig=save_progress_figs, project_dir=project_dir)

        if save_every_n_iters>0 and (iteration%save_every_n_iters)==0:
            save_checkpoint(model, data, history, batch_info, iteration, name=name,
                            project_dir=project_dir,save_history=save_history, 
                            save_states=save_states, save_data=save_data)
            
        try: model = resample_model(data, **model, ar_only=ar_only)
        except KeyboardInterrupt: break
    
    return model, history, name
    
    
def resume_fitting(*, params, hypparams, batch_info, iteration, mask,
                   conf, Y, seed, noise_prior=None, states=None, **kwargs):
    
    model = initialize_model(
        states=states, params=params, hypparams=hypparams,
        noise_prior=noise_prior, seed=seed, Y=Y, mask=mask,
        conf=conf, **kwargs)
    
    data = jax.device_put({'Y':Y, 'conf':conf, 'mask':mask})
    
    return fit_model(model, data, batch_info, start_iter=iteration+1, **kwargs)


def apply_model(*, params, coordinates, confidences=None,
                num_iters=5, use_saved_states=True, states=None, 
                mask=None, batch_info=None, ar_only=False, 
                random_seed=0, batch_length=None, save_results=True,
                project_dir=None, name=None, results_path=None, 
                Y=None, conf=None, noise_prior=None, return_model_only=False, **kwargs):   
    
    
    data,new_batch_info = format_data(
        coordinates, confidences=confidences, batch_length=None, **kwargs)
    session_names = [key for key,start,end in new_batch_info]

    # SR: Debug
    print(data.keys())
    print(session_names)

    if save_results:
        if results_path is None: 
            assert project_dir is not None and name is not None, fill(
                'The ``save_results`` option requires either a ``results_path`` '
                'or the ``project_dir`` and ``name`` arguments')
            results_path = os.path.join(project_dir,name,'results.h5')
     
    if use_saved_states:
        assert not (states is None or mask is None or batch_info is None), fill(
            'The ``use_saved_states`` option requires the additional '
            'arguments ``states``, ``mask`` and ``batch_info``')   
        
        new_states = {}
        for k,v in jax.device_get(states).items():
            new_states[k] = batch(
                unbatch(v, mask, batch_info), 
                keys=session_names)[0]
        states = new_states
    else: states = None
    
    model = initialize_model(
        states=states, params=params, 
        **jax.device_put(data), **kwargs)
    
    if num_iters>0:
        for iteration in tqdm.trange(num_iters, desc='Applying model'):
            model = resample_model(data, **model, ar_only=ar_only, states_only=True)
        
    nlags = model['hypparams']['ar_hypparams']['nlags']
    states = jax.device_get(model['states'])                     
    estimated_coords = jax.device_get(estimate_coordinates(
        **model['states'], **model['params'], **data))
    
    mask = np.array(data['mask'])
    usage = get_usages(states['z'], mask)
    reindex = np.argsort(np.argsort(usage)[::-1])
    z_reindexed = reindex[states['z']]
    
    results_dict = {
        session_name : {
            'syllables' : np.pad(states['z'][i], (nlags,0), mode='edge')[m>0],
            'syllables_reindexed' : np.pad(z_reindexed[i], (nlags,0), mode='edge')[m>0],
            'estimated_coordinates' : estimated_coords[i][m>0],
            'latent_state' : states['x'][i][m>0],
            'centroid' : states['v'][i][m>0],
            'heading' : states['h'][i][m>0],
        } for i,(m,session_name) in enumerate(zip(mask,session_names))}
    
    if save_results: 
        save_hdf5(results_path, results_dict)
        print(fill(f'Saved results to {results_path}'))

    if return_model_only:
        return model, data
        
    return results_dict


def revert(checkpoint, iteration):
    
    assert len(checkpoint['history'])>0, fill(
        'No history was saved during fitting')
    
    use_iter = max([i for i in checkpoint['history'] if i <= iteration])
    print(f'Reverting to iteration {use_iter}')
    
    model_snapshot =  checkpoint['history'][use_iter]
    checkpoint['params'] = model_snapshot['params']
    checkpoint['seed'] = model_snapshot['seed']
    checkpoint['iteration'] = use_iter
    
    if 'states' in model_snapshot: 
        checkpoint['states'] = model_snapshot['states']
    else: checkpoint['states'] = None
        
    for i in list(checkpoint['history'].keys()):
        if i > use_iter: del checkpoint['history'][i]

    return checkpoint
    
    
def update_hypparams(model_dict, **kwargs):
    
    assert 'hypparams' in model_dict, fill(
        'The inputted model/checkpoint does not contain any hyperparams')
    
    not_updated = list(kwargs.keys())
    
    for hypparms_group in model_dict['hypparams']:
        for k,v in kwargs.items():
            
            if k in model_dict['hypparams'][hypparms_group]:
                
                old_value = model_dict['hypparams'][hypparms_group][k]
                
                if not np.isscalar(old_value): print(fill(
                    f'{k} cannot be updated since it is not a scalar hyperparam'))
                 
                else:
                    if not isinstance(v, type(old_value)): warnings.warn(fill(
                        f'{v} has type {type(v)} which differs from the current '
                        f'value of {k} which has type {type(old_value)}. {v} will '
                        f'will be cast to {type(old_value)}'))
                                     
                    model_dict['hypparams'][hypparms_group][k] = type(old_value)(v)
                    not_updated.remove(k)
                    
    if len(not_updated)>0: warnings.warn(fill(
        f'The following hypparams were not found {not_updated}'))
        
    return model_dict

