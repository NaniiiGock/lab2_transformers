#!/bin/bash

#SBATCH -J calculate_bleu           
#SBATCH -N 1                        
#SBATCH --ntasks=1                  
#SBATCH -t 01:00:00                 
#SBATCH --mem=32G                   
#SBATCH --partition=gpu            
#SBATCH --gres=gpu:tesla:1        

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Environment modules loaded:"
module list

echo "Generating translations..."
fairseq-generate sentencepiece \
    --path checkpoints/checkpoint_best.pt \
    --source-lang et \
    --target-lang en \
    --batch-size 64 \
    --beam 5 \
    --remove-bpe \
    --gen-subset test \
    --task translation > test_output.txt

echo "Extracting hypotheses..."
grep ^H test_output.txt | sort -V | cut -f3- > hypothesis.en

# echo "Detokenizing hypotheses..."
# python3 - <<EOF
# import sentencepiece as spm

# # Load SentencePiece model
# sp = spm.SentencePieceProcessor(model_file="sentencepiece/sentencepiece.model")

# # Detokenize each line in the hypotheses file
# with open("hypothesis.en", "r", encoding="utf-8") as fin, open("hypothesis.detok.en", "w", encoding="utf-8") as fout:
#     for line in fin:
#         detokenized = sp.decode(line.strip().split())
#         fout.write(detokenized + "\n")
# EOF

num_hypotheses=$(wc -l < hypothesis.detok.en)
num_references=$(wc -l < sentencepiece/test.sp.en)

if [ "$num_hypotheses" -ne "$num_references" ]; then
    echo "Error: Mismatched line counts!"
    echo "Hypotheses: $num_hypotheses, References: $num_references"
    echo "Trimming references to match hypotheses..."
    head -n $num_hypotheses sentencepiece/test.sp.en > trimmed_test.sp.en
    reference_file="trimmed_test.sp.en"
else
    reference_file="sentencepiece/test.sp.en"
fi

echo "Calculating BLEU score..."
cat hypothesis.en | sacrebleu $reference_file > bleu_score.txt

echo "BLEU score:"
cat bleu_score.txt

echo "Process complete!"
