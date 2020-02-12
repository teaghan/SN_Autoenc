#!/bin/bash

module load python/3.6
source $HOME/torchmed/bin/activate
cp /scratch/obriaint/SN_Autoenc/data/mean_and_std_PAYNE_specs.npy $SLURM_TMPDIR
cp /scratch/obriaint/SN_Autoenc/data/normalization_data.npz $SLURM_TMPDIR
cp /scratch/obriaint/SN_Autoenc/data/mock_all_spectra_no_noise_resample_prior_large.npz $SLURM_TMPDIR
python /scratch/obriaint/SN_Autoenc/train_synth_ae.py ae_1 -v 1000 -dd $SLURM_TMPDIR/
cp /scratch/obriaint/SN_Autoenc/data/aspcapStar_dr14.h5 $SLURM_TMPDIR
python /scratch/obriaint/SN_Autoenc/train_obs_ae.py ae_1 -v 1000 -dd $SLURM_TMPDIR/
