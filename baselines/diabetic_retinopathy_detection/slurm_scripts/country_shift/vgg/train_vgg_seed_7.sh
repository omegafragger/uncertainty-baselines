#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2g.20gb:1

#SBATCH --job-name="train_retino_country_shift/vgg"

source activate ub

srun python torch_deterministic.py \
--output-dir /scratch-ssd/jisoti/ub_retinopathy_outputs/country_shift/vgg \
--data-dir /scratch-ssd/jisoti/ub_retinopathy \
--distribution-shift aptos \
--seed 7 \
--per-core-batch-size 32 \
--model vgg16
