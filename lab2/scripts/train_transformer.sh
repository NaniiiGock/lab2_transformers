#!/bin/bash

#SBATCH -J preprocess_translation        
#SBATCH -N 1                             
#SBATCH --ntasks=1                       
#SBATCH -t 01:00:00                     
#SBATCH --mem=32G                         
#SBATCH --partition=gpu                 
#SBATCH --gres=gpu:tesla:1

conda activate transformers-course

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Environment modules loaded:"
module list

fairseq-train data-bin/sp-translation \
    --arch transformer \
    --encoder-layers 6 \
    --decoder-layers 6 \
    --encoder-embed-dim 256 \
    --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 1024 \
    --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-tokens 12000 \
    --share-all-embeddings \
    --max-epoch 20 \
    --save-dir checkpoints \
    --fp16 \
    --restore-file checkpoints/checkpoint_8_*.pt

echo "Training completed!"