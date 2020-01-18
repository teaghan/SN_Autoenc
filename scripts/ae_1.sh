#!/bin/bash

module load python/3.6
source $HOME/torchmed/bin/activate
cp /scratch/obriaint/Cycle_SN/data/mean_and_std_PAYNE_specs.npy $SLURM_TMPDIR
cp /scratch/obriaint/Cycle_SN/data/mock_all_spectra_no_noise_resample_prior_large.npz $SLURM_TMPDIR
python /scratch/obriaint/AN_autenc/train_network.py train_synth_ae.py ae_1 -v 1000 -dd $SLURM_TMPDIR/
