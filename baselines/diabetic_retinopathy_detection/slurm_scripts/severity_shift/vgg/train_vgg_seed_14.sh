#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name="train_retino_sev_shift/vgg"

source activate ub2

srun python torch_deterministic.py \
--output-dir /scratch-ssd/jisoti/ub_retinopathy_outputs/sev_shift/vgg \
--data-dir /users/jisoti/ub_retinopathy \
--distribution-shift severity \
--seed 14 \
--per-core-batch-size 32 \
--model vgg16
