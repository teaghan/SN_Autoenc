# for new1, reconstruct gradients
import numpy as np
import h5py
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def vec_bin_array(arr, m):
    """
    Arguments: 
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret 

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Different data directory from sn_autoencoder/data/ dir.", 
                        type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def weighted_masked_mse_loss(pred, target, error, mask):
    '''
    Mean-squared-error weighted by the error on the target 
    and using a mask for the bad pixels in the target.
    '''
    return torch.mean(((pred - target)*mask/error) ** 2)
    #return torch.mean(((pred - target)/error) ** 2)

def create_synth_batch(model, x_mean, x_std, y=None, batchsize=8,
                       labels_payne=None, perturbations=None):
    if y is None:
        # Collect a batch of stellar parameters from the Payne training distribution
        y = np.copy(labels_payne[np.random.randint(len(labels_payne), size=batchsize)])

        # Apply random perturbation within limits
        y += np.array([np.random.uniform(-1*p, p, size=batchsize) for p in perturbations]).T
    
        # Clip minimum Vturb, C12/C13, and Vmacro values
        for i in [2,23,24]:
            y[y[:,i]<np.min(labels_payne[:,i]),i] = np.min(labels_payne[:,i])
        
        y = Variable(torch.Tensor(y.astype(np.float32)))

    # Create a batch of synthetic spectra
    x = model.run_emulator(y)

    # Normalize the spectra
    x = (x - x_mean) / x_std
    
    # Only select last 7167 pixels
    x = x[:,47:]
    
    return {'x':x, 'x_err':torch.ones(x.size(), dtype=torch.float32), 
            'x_msk':torch.ones(x.size(), dtype=torch.float32), 'y':y} 
    
def batch_to_cuda(batch):
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    return batch

class PayneObservedDataset(Dataset):
    
    """
            
    """

    def __init__(self, data_file_obs, obs_domain, dataset='train', x_mean=1., x_std=1., 
                 collect_x_mask=False):
        
        self.data_file_obs = data_file_obs
        self.obs_domain = obs_domain
        self.dataset = dataset
        self.x_mean = x_mean
        self.x_std = x_std
        self.collect_x_mask = collect_x_mask
        
    def __len__(self):
        with h5py.File(self.data_file_obs, "r") as F_obs:
            num_samples =  len(F_obs[self.obs_domain+' spectrum '+self.dataset])
        return num_samples
    
    def __getitem__(self, idx, dataset=None, return_labels=False, collect_preceeding=False):
        if dataset==None:
            dataset=self.dataset
        
        with h5py.File(self.data_file_obs, "r") as F_obs: 
            if collect_preceeding:
                # Collect all samples before idx
                x = torch.from_numpy(F_obs[self.obs_domain + ' spectrum ' + dataset][:idx,47:].astype(np.float32))
                x_err = torch.from_numpy(F_obs[self.obs_domain + ' error_spectrum ' + dataset][:idx,47:].astype(np.float32))
                if self.collect_x_mask:
                    x_msk = torch.from_numpy(F_obs[self.obs_domain+' bit_mask_spectrum '+dataset][:idx,47:].astype(np.float32))
                else:
                    x_msk = torch.ones(x.size(), dtype=torch.float32)
                if return_labels:
                    y = torch.from_numpy(F_obs[self.obs_domain + ' labels ' + dataset][:idx].astype(np.float32)) 
            else:
                # Collect all samples before idx
                x = torch.from_numpy(F_obs[self.obs_domain + ' spectrum ' + dataset][idx,47:].astype(np.float32))
                x_err = torch.from_numpy(F_obs[self.obs_domain + ' error_spectrum ' + dataset][idx,47:].astype(np.float32))
                if self.collect_x_mask:
                    x_msk = torch.from_numpy(F_obs[self.obs_domain+' bit_mask_spectrum '+dataset][idx,47:].astype(np.float32))
                else:
                    x_msk = torch.from_numpy(np.ones(x.shape).astype(np.float32))
                if return_labels:
                    y = torch.from_numpy(F_obs[self.obs_domain + ' labels ' + dataset][idx].astype(np.float32)) 
                    
            # Normalize the spectra
            x = (x - self.x_mean) / self.x_std
            
            # Add one to the spectra errors to ensure that the minimum
            # error is 1. This helps avoid huge losses.
            x_err += 1
        if return_labels:    
            return {'x':Variable(x), 'x_err':x_err, 'x_msk':x_msk, 'y':Variable(y)} 
        else:    
            return {'x':x, 'x_err':x_err, 'x_msk':x_msk} 
        
def evaluate_dxdy(synth_ae, synth_train_batch, x_mean, x_std, 
                  dy_batch, dxdy_mean_batch, dxdy_std_batch,  use_cuda):

    # Repeat labels into matrix
    y_ref = torch.cat(synth_train_batch['y'].size(1)*[synth_train_batch['y'].clone()])
    # Apply + an - deviations 
    y_pos = y_ref.clone()+dy_batch
    y_neg = y_ref.clone()-dy_batch

    # Produce x_synth using the emulator for + and - 
    x_pos = synth_ae.run_emulator(y_pos)
    x_neg = synth_ae.run_emulator(y_neg)

    # Normalize the spectra
    x_pos = (x_pos - x_mean) / x_std
    x_neg = (x_neg - x_mean) / x_std
    
    # Only select last 7167 pixels
    x_pos = x_pos[:,47:]
    x_neg = x_neg[:,47:]
    
    # Calculate the Jacobian for the emulator
    J_tgt = x_pos - x_neg
    
    # Produce x_synth using the decoder for + and -
    x_pos = synth_ae.yz_to_xsynth(y_pos)
    x_neg = synth_ae.yz_to_xsynth(y_neg)
    
    # Calculate Jacobian for the decoder
    J_dec = x_pos - x_neg
    
    # Normalize to give each label equal weighting
    J_tgt = (J_tgt - dxdy_mean_batch)/dxdy_std_batch
    J_dec = (J_dec - dxdy_mean_batch)/dxdy_std_batch

    j_loss = torch.mean((J_tgt.detach()-J_dec)**2)
    
    return j_loss
        
def train_synth_iter(synth_ae, synth_train_batch, x_loss_fn, y_loss_fn, 
                     loss_weight_x, loss_weight_y, loss_weight_dxdy, 
                     optimizer, lr_scheduler, losses_cp, cur_iter, 
                     x_mean, x_std, dy_batch, dxdy_mean_batch, dxdy_std_batch, use_cuda,
                    rec_grads):
    
    # RUN MODEL
    
    synth_ae.train_mode()
    # Encoding
    y_synth, z_synth = synth_ae.x_to_yz(synth_train_batch['x'].detach())
    # Decoding
    x_synthsynth = synth_ae.yz_to_xsynth(synth_train_batch['y'].detach())

    # EVALUATE LOSSES
    
    # Targets for y_synth
    y_tgt = (synth_train_batch['y'] - synth_ae.y_min)/(synth_ae.y_max - synth_ae.y_min) - 0.5
    # Targets for z_synth
    z_tgt = torch.zeros((len(synth_train_batch['y']), synth_ae.num_z), dtype=torch.float32)
    
    # x to y
    y_loss = y_loss_fn(y_synth, y_tgt)
    # x to z
    z_loss = y_loss_fn(z_synth, z_tgt)
    # y to x
    xsynth_loss = x_loss_fn(pred=x_synthsynth, 
                            target=synth_train_batch['x'], 
                            error=synth_train_batch['x_err'], 
                            mask=synth_train_batch['x_msk'])
    
    # Evaluate the dx/dy Jacobians and compare them to the the Jacobian produced by the emulator
    if rec_grads:
        loss_grads_synth = evaluate_dxdy(synth_ae, synth_train_batch, x_mean, x_std, 
                                         dy_batch, dxdy_mean_batch, dxdy_std_batch,  use_cuda)
    else:
        loss_grads_synth = torch.tensor(np.array([0]).astype(np.float32))
    
    # Combine losses with appropriate loss weights
    loss_total = (loss_weight_y * y_loss + 
                  loss_weight_y * z_loss +
                  loss_weight_x * xsynth_loss +
                  loss_weight_dxdy * loss_grads_synth)

    # Back propogate
    optimizer.zero_grad()
    loss_total.backward()
    # Adjust network weights
    optimizer.step()    
    # Adjust learning rate
    lr_scheduler.step()

    # Save losses
    losses_cp['y_synth'].append(y_loss.data.item())
    losses_cp['z_synth'].append(z_loss.data.item())
    losses_cp['x_synth'].append(xsynth_loss.data.item())
    losses_cp['dxdy_synth'].append(loss_grads_synth.data.item())

    return losses_cp
'''
def compute_dydx_jacobian(ae, x, num_y=25):
    # Batchsize
    b = x.shape[0]
    # Shape of spectra
    x_shape = x.shape[1:]
    # Add dimension
    x = x.unsqueeze(1)
    # Repeat spectrum along new axis for each label 
    x = x.repeat(1, num_y, *(1,)*len(x.shape[2:]))
    x.requires_grad_(True)
    tmp_shape = x.shape
    # Forward pass
    y, z = ae.x_to_yz(x.reshape(-1, *tmp_shape[2:]))
    y_shape = y.shape[1:]
    y = y.reshape(b, num_y, num_y)
    # Create a unit vector for each label
    input_val = torch.eye(num_y).reshape(1, num_y, num_y).repeat(b, 1, 1)
    # Backward pass
    y.backward(input_val)
    # Return jacobian
    return x.grad.reshape(b, *y_shape, *x_shape)

I don't think the above fcn is differentiable, might need to try:
'''

def compute_dydx_jacobian(ae, x, num_y=25):
    # Batchsize
    b = x.shape[0]
    # Shape of spectra
    x_shape = x.shape[1:]
    # Add dimension
    x = x.unsqueeze(1)
    # Repeat spectrum along new axis for each label 
    x = x.repeat(1, num_y, *(1,)*len(x.shape[2:]))
    x.requires_grad_(True)
    tmp_shape = x.shape
    # Forward pass
    y, _ = ae.x_to_yz(x.reshape(-1, *tmp_shape[2:]))
    y_shape = y.shape[1:]
    y = y.reshape(b, num_y, num_y)
    # Create a unit vector for each label
    input_val = torch.eye(num_y).reshape(1, num_y, num_y).repeat(b, 1, 1)
    
    # Backward pass
    jacobian = torch.autograd.grad(y, x,
                                   grad_outputs=input_val,
                                   retain_graph=True,
                                   create_graph=True,
                                   allow_unused=True)[0].requires_grad_()
    
    # Return jacobian
    return jacobian.reshape(b, *y_shape, *x_shape)

def evaluate_dydx_obs(synth_ae, obs_ae, synth_train_batch, x_mean, x_std, 
                       dydx_mean, dydx_std, use_cuda):
    
    # Calculate the dy/dx Jacobians
    J_tgt = compute_dydx_jacobian(synth_ae, synth_train_batch['x'].detach())
    J_enc = compute_dydx_jacobian(obs_ae, synth_train_batch['x'].detach())
        
    # Normalize to give each label equal weighting
    J_tgt = (J_tgt - dydx_mean)/dydx_std
    J_enc = (J_enc - dydx_mean)/dydx_std

    j_loss = torch.mean((J_tgt.detach()-J_enc)**2)
    
    return j_loss

def evaluate_dxdy_obs(synth_ae, obs_ae, synth_train_batch, x_mean, x_std, 
                           dy_batch, dxdy_mean_batch, dxdy_std_batch,  use_cuda):

    # Repeat labels into matrix
    y_ref = torch.cat(synth_train_batch['y'].size(1)*[synth_train_batch['y'].clone()])
    # Apply + an - deviations 
    y_pos = y_ref.clone()+dy_batch
    y_neg = y_ref.clone()-dy_batch

    # Produce x_synth using the emulator for + and - 
    x_pos = synth_ae.run_emulator(y_pos)
    x_neg = synth_ae.run_emulator(y_neg)

    # Normalize the spectra
    x_pos = (x_pos - x_mean) / x_std
    x_neg = (x_neg - x_mean) / x_std
    
    # Only select last 7167 pixels
    x_pos = x_pos[:,47:]
    x_neg = x_neg[:,47:]
    
    # Calculate the Jacobian for the emulator
    J_tgt = x_pos - x_neg
    
    # Produce x_obs using the decoder for + and -
    x_pos = obs_ae.yz_to_xsynth(y_pos)
    x_neg = obs_ae.yz_to_xsynth(y_neg)
    
    # Calculate Jacobian for the decoder
    J_dec = x_pos - x_neg
    
    # Normalize to give each label equal weighting
    J_tgt = (J_tgt - dxdy_mean_batch)/dxdy_std_batch
    J_dec = (J_dec - dxdy_mean_batch)/dxdy_std_batch

    j_loss = torch.mean((J_tgt.detach()-J_dec)**2)
    
    return j_loss

def train_obs_iter(synth_ae, obs_ae, synth_train_batch, obs_train_batch, x_loss_fn, 
                   loss_weight_x, loss_weight_dxdy, loss_weight_dydx, optimizer, lr_scheduler, losses_cp, 
                   cur_iter, x_mean, x_std, dy_batch, dxdy_mean_batch, dxdy_std_batch, 
                   dydx_mean, dydx_std, use_cuda, rec_grads):
    
    # RUN MODEL
    
    obs_ae.train_mode()
    synth_ae.eval_mode()
    # Encoding
    y_obs, z_obs = obs_ae.x_to_yz(obs_train_batch['x'].detach())
    # Decoding
    x_obsobs = obs_ae.yz_to_xobs(y_obs, z_obs)

    # EVALUATE LOSSES
    
    # x to x
    xobs_loss = x_loss_fn(pred=x_obsobs, 
                            target=obs_train_batch['x'], 
                            error=obs_train_batch['x_err'], 
                            mask=obs_train_batch['x_msk'])
    
    # Evaluate the dx/dy Jacobians and compare them to the the Jacobian produced by the emulator
    if rec_grads:
        loss_dxdy_obs = evaluate_dxdy_obs(synth_ae, obs_ae, synth_train_batch, 
                                                x_mean, x_std, dy_batch, dxdy_mean_batch, 
                                                dxdy_std_batch,  use_cuda)
        loss_dydx_obs = evaluate_dydx_obs(synth_ae, obs_ae, synth_train_batch, 
                                          x_mean, x_std, dydx_mean, 
                                          dydx_std,  use_cuda)
    else:
        loss_dxdy_obs = torch.tensor(np.array([0]).astype(np.float32))
        loss_dydx_obs = torch.tensor(np.array([0]).astype(np.float32))
            
    # Combine losses with appropriate loss weights
    loss_total = (loss_weight_x * xobs_loss +
                  loss_weight_dxdy * loss_dxdy_obs +
                  loss_weight_dydx * loss_dydx_obs)

    
    # Back propogate
    optimizer.zero_grad()
    loss_total.backward()
    # Adjust network weights
    optimizer.step()
    # Adjust learning rate
    lr_scheduler.step()

    # Save losses
    losses_cp['x_obs'].append(xobs_loss.data.item())
    losses_cp['dxdy_obs'].append(loss_dxdy_obs.data.item())
    losses_cp['dydx_obs'].append(loss_dydx_obs.data.item())
                      
    return losses_cp

def train_dydx_iter(synth_ae, obs_ae, synth_train_batch, x_loss_fn, 
                   loss_weight_x, loss_weight_j, optimizer, lr_scheduler, losses_cp, 
                   cur_iter, x_mean, x_std, dy_batch, dxdy_mean_batch, dxdy_std_batch, 
                   dydx_mean, dydx_std, use_cuda, rec_grads):
    
    # RUN MODEL
    
    obs_ae.train_mode()
    synth_ae.eval_mode()
    
    # Evaluate the dx/dy Jacobians and compare them to the the Jacobian produced by the emulator
    if rec_grads:
        loss_dydx_obs = evaluate_dydx_obs(synth_ae, obs_ae, synth_train_batch, 
                                          x_mean, x_std, dydx_mean, 
                                          dydx_std,  use_cuda)
    else:
        loss_dydx_obs = torch.tensor(np.array([0]).astype(np.float32))
            
    # Combine losses with appropriate loss weights
    loss_total = (loss_weight_j * loss_dydx_obs)

    
    # Back propogate
    optimizer.zero_grad()
    loss_total.backward()
    # Adjust network weights
    optimizer.step()
    # Adjust learning rate
    lr_scheduler.step()

    # Save losses
    losses_cp['dydx_obs'].append(loss_dydx_obs.data.item())
                      
    return losses_cp

def evaluation_checkpoint(obs_ae, obs_val_set, y_loss_fn, x_loss_fn, losses_cp):
    # Evaluate validation set
    obs_ae.eval_mode()

     # Encoding
    y_obs, z_obs = obs_ae.x_to_yz(obs_val_set['x'].detach())
    # Decoding
    x_obsobs = obs_ae.yz_to_xobs(obs_val_set['y'].detach(), z_obs, raw_labels=True)

    # EVALUATE LOSSES
    
    # Targets for y_obs
    y_tgt = (obs_val_set['y'] - obs_ae.y_min)/(obs_ae.y_max - obs_ae.y_min) - 0.5
    
    # x to y
    y_loss = y_loss_fn(y_obs, y_tgt)
    # y to x
    xobs_loss = x_loss_fn(pred=x_obsobs, 
                          target=obs_val_set['x'], 
                          error=obs_val_set['x_err'], 
                          mask=obs_val_set['x_msk'])

    # Save losses
    losses_cp['y_obs_val'].append(y_loss.data.item())
    losses_cp['x_obs_val'].append(xobs_loss.data.item())
    
    return losses_cp
