#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name="train_retino_sev_shift/vgg"

source activate ub

srun python torch_deterministic.py \
--output-dir /scratch-ssd/jisoti/ub_retinopathy_outputs/sev_shift/vgg \
--data-dir /scratch-ssd/jisoti/ub_retinopathy \
--distribution-shift severity \
--seed 12 \
--per-core-batch-size 32 \
--model vgg16
