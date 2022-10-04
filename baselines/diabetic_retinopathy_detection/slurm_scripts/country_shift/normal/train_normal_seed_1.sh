#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

#SBATCH --job-name="train_retino_country_shift/normal"

source activate ub

srun python torch_deterministic.py \
--output-dir /scratch-ssd/jisoti/ub_retinopathy_outputs/country_shift/normal \
--data-dir /scratch-ssd/jisoti/ub_retinopathy \
--distribution-shift aptos \
--seed 1 \
--per-core-batch-size 32 \
--model resnet50
