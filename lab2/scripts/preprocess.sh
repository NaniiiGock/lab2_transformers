#!/bin/bash

#SBATCH -J preprocess_translation        
#SBATCH -N 1                           
#SBATCH --ntasks=1                      
#SBATCH -t 01:00:00                     
#SBATCH --mem=8G                         
#SBATCH --partition=gpu                 

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Environment modules loaded:"
module list

fairseq-preprocess \
    --source-lang en \
    --target-lang et \
    --trainpref sentencepiece/train.sp \
    --validpref sentencepiece/dev.sp \
    --testpref sentencepiece/test.sp \
    --destdir data-bin/sp-translation \
    --joined-dictionary \
    --workers 4


echo "Preprocessing complete!"
