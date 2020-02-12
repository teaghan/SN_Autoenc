from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from collections import defaultdict
import os
import sys
import time
import configparser

import torch
from torch.utils.data import DataLoader

from training_fns import (parseArguments, weighted_masked_mse_loss, 
                          create_synth_batch, batch_to_cuda, train_synth_iter)
from network_new import SN_AE

np.random.seed(1)
torch.manual_seed(1)

# Check for GPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU!')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(1)
else:
    torch.set_default_tensor_type('torch.FloatTensor')
                    
# Collect the command line arguments

args = parseArguments()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir
'''
model_name = 'ae_1'
verbose_iters = 10
cp_time = 15
data_dir = 'data/'
'''
# Directories

cur_dir = os.path.dirname(__file__)
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
progress_dir = os.path.join(cur_dir, 'progress/')

if args.data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')
architecture_config = config['ARCHITECTURE']
print('\nCreating model: %s'%model_name)
print('\nConfiguration:')
for key_head in config.keys():
    if key_head=='DEFAULT':
        continue
    print('  %s' % key_head)
    for key in config[key_head].keys():
        print('    %s: %s'%(key, config[key_head][key]))
        
# DATA FILES
data_file_obs = os.path.join(data_dir, config['DATA']['data_file_obs'])
norm_file = os.path.join(data_dir, config['DATA']['norm_file'])
emulator_fn = os.path.join(model_dir, config['DATA']['emulator_fn'])

# TRAINING PARAMETERS
batchsize = int(config['TRAINING']['batchsize'])
learning_rate_encoder = float(config['TRAINING']['learning_rate_encoder'])
learning_rate_decoder = float(config['TRAINING']['learning_rate_decoder'])
loss_weight_x = float(config['TRAINING']['loss_weight_x_synth'])
loss_weight_y = float(config['TRAINING']['loss_weight_y'])
loss_weight_j = float(config['TRAINING']['loss_weight_j_synth'])
total_synth_batch_iters = float(config['TRAINING']['total_synth_batch_iters'])
total_obs_batch_iters = float(config['TRAINING']['total_obs_batch_iters'])
lr_decay_batch_iters = eval(config['TRAINING']['lr_decay_batch_iters'])
lr_decay = float(config['TRAINING']['lr_decay'])
            

# BUILD THE NETWORKS

# Construct the Auto-encoder
print('\nBuilding networks...')
synth_ae = SN_AE(architecture_config, emulator_fn, use_cuda=use_cuda)

# Display model architectures
print('\n\nEMULATOR ARCHITECTURE:\n')
print(synth_ae.emulator)
print('\n\nENCODER ARCHITECTURE:\n')
print(synth_ae.encoder)
print('\n\nDECODER ARCHITECTURE:\n')
print(synth_ae.decoder)

# Construct optimizer
optimizer = torch.optim.Adam([{'params': synth_ae.encoder.parameters(), "lr": learning_rate_encoder},
                                  {'params': synth_ae.decoder.parameters(), "lr": learning_rate_decoder}],
                                 weight_decay = 0, betas=(0.5, 0.999))
# Learning rate decay
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=lr_decay_batch_iters, 
                                                    gamma=lr_decay)

# Loss functions
y_loss_fn = torch.nn.MSELoss()
x_loss_fn = weighted_masked_mse_loss

# Check for pre-trained weights
model_filename =  os.path.join(model_dir,model_name+'_synth.pth.tar')
if os.path.exists(model_filename):
    fresh_model = False
else:
    fresh_model = True
    
# Load pretrained model
if fresh_model:
    print('\nStarting fresh model to train...')
    cur_iter = 1
    losses = defaultdict(list)
else:
    print('\nLoading saved model to continue training...')
    # Load model info
    checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
    cur_iter = checkpoint['batch_iters']+1
    losses = dict(checkpoint['losses'])
    
    if cur_iter>=total_synth_batch_iters:
        print('Training already complete.')
        sys.exit()
    
    # Load optimizer states
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    # Load model weights
    synth_ae.load_state_dict(checkpoint['synth_ae'])
    
# Normalization data for the spectra and Jacobians
norm_data = np.load(norm_file)
x_mean = torch.Tensor(norm_data['x_mean'].astype(np.float32))
x_std = torch.Tensor(norm_data['x_std'].astype(np.float32))
# Used to re-scale the Jacobians for dx/dy in order to give each label 
# roughly equal weighting.
dydx_mean = torch.Tensor(norm_data['dydx_mean'].astype(np.float32))
dydx_std = torch.Tensor(norm_data['dydx_std'].astype(np.float32))
dxdy_mean = torch.Tensor(norm_data['dxdy_mean'].astype(np.float32))
dxdy_std = torch.Tensor(norm_data['dxdy_std'].astype(np.float32))

# Load the Payne labels

# [ Teff, Logg, Vturb [km/s],
# [C/H], [N/H], [O/H], [Na/H], [Mg/H],
# [Al/H], [Si/H], [P/H], [S/H], [K/H],
# [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],
# [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H],
# C12/C13, Vmacro [km/s] ]
labels_payne = np.load(data_dir+'mock_all_spectra_no_noise_resample_prior_large.npz')['labels'].T

# Perturb the payne labels within a range to create our synthetic training batches.
# These perturbations are in the same order as the labels.
perturbations = [100., 0.1, 0.2, *np.repeat(0.1, 20), 5., 2.]

# Calculate the matrix that will be applied to our "reference" labels
# to evaluate our gradients throughout training.
dy = torch.Tensor(np.array([[25., 0.025, 0.05, *np.repeat(0.025, 20), 1.25, 0.5]]).astype(np.float32))


# Turn these into matrices that can easily be applied to a batch.
dy_batch = torch.zeros((batchsize*dy.size(1), dy.size(1)))
dxdy_mean_batch = torch.zeros((batchsize*dy.size(1), 1))
dxdy_std_batch = torch.zeros((batchsize*dy.size(1), 1))
for i, indx in enumerate(range(0, dy_batch.size(0), batchsize)):
    dy_batch[indx:indx+batchsize, i] = dy[0,i]
    dxdy_mean_batch[indx:indx+batchsize] = dxdy_mean[i]
    dxdy_std_batch[indx:indx+batchsize] = dxdy_std[i]
    
def train_network(cur_iter):
    print('Training the network...')
    print('Progress will be displayed every %i iterations and the model will be saved every %i minutes.'%
          (verbose_iters,cp_time))
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    
    while cur_iter < total_synth_batch_iters:
        
        # Create synthetic batch from the distribution of the original Payne training set.
        synth_train_batch = create_synth_batch(synth_ae, x_mean, x_std, 
                                               batchsize=batchsize,
                                               labels_payne=labels_payne, 
                                               perturbations=perturbations)

        # Switch to GPU
        if use_cuda:
            synth_train_batch = batch_to_cuda(synth_train_batch)

        # Train an iteration
        losses_cp = train_synth_iter(synth_ae, synth_train_batch, x_loss_fn, y_loss_fn, 
                                     loss_weight_x, loss_weight_y, loss_weight_j, 
                                     optimizer, lr_scheduler, losses_cp, cur_iter, 
                                     x_mean, x_std, dy_batch, dxdy_mean_batch, dxdy_std_batch, use_cuda,
                                     rec_grads=True)

        # Display losses
        if cur_iter % verbose_iters == 0:

            # Calculate averages
            for k in losses_cp.keys():
                losses[k].append(np.mean(losses_cp[k]))
            losses['batch_iters'].append(cur_iter)

            # Print current status
            print('\nBatch Iterations: %i/%i ' % (cur_iter, total_synth_batch_iters))
            print('Training Losses:')
            print('\t|   Y_loss   |    X_loss   |   Z_loss   | dXdY_loss |')
            print('Synth   |   %0.5f  |   %0.5f   |   %0.5f  |  %0.5f  |' % 
                  (losses['y_synth'][-1], losses['x_synth'][-1], losses['z_synth'][-1], 
                   losses['dxdy_synth'][-1]))

            print('\n') 

            # Save losses to file to analyze throughout training. 
            np.save(os.path.join(progress_dir, model_name+'_losses.npy'), losses) 
            # Reset checkpoint loss dict
            losses_cp = defaultdict(list)
            # Free some GPU memory
            torch.cuda.empty_cache()

        # Increase the iteration
        cur_iter += 1

        # Save periodically
        if time.time() - cp_start_time >= cp_time*60:
            print('Saving network...')

            torch.save({'batch_iters': cur_iter,
                        'losses': losses,
                        'optimizer' : optimizer.state_dict(),
                        'lr_scheduler' : lr_scheduler.state_dict(),
                        'synth_ae' : synth_ae.state_dict()}, 
                        model_filename)

            cp_start_time = time.time()

        if cur_iter>total_synth_batch_iters:
            break
    torch.save({'batch_iters': cur_iter,
                'losses': losses,
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'synth_ae' : synth_ae.state_dict()}, 
               model_filename)
    
# Run the training
if __name__=="__main__":
    train_network(cur_iter)
    print('\nTraining complete.')
