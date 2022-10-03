import argparse
import datetime
import os
import pathlib
import pprint
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import uncertainty_baselines as ub
import utils  # local file import
import wandb

from tensorboard.plugins.hparams import api as hp

DEFAULT_NUM_EPOCHS = 90

def training_args():
    output_dir = './'
    data_dir = './'

    preproc_builder_config = 'btgraham-300'
    dr_decision_threshold = 'moderate'
    
    load_from_checkpoint = False
    checkpoint_dir = None
    cache_eval_datasets = False
    
    use_wandb = False
    wandb_dir = 'wandb'
    project = 'ub-debug'
    exp_name = None
    exp_group = None
    
    distribution_shift = 'aptos'
    load_train_split = True
    
    base_learning_rate = 0.023072
    final_decay_factor = 0.01
    one_minus_momentum = 0.0098467
    lr_schedule = 'step'
    lr_warmup_epochs = 1
    lr_decay_ratio = 0.2
    lr_decay_epoch_1 = 30
    lr_decay_epoch_2 = 60
    
    seed = 42
    class_reweight_mode = None
    l2 = 0.00010674
    train_epochs = DEFAULT_NUM_EPOCHS
    per_core_batch_size = 32
    checkpoint_interval = 25
    num_bins = 15
    force_use_cpu = False
    use_gpu = True
    use_bfloat16 = False
    num_cores = 1
    tpu = None
    
    model = "resnet50"
    sn_coeff = 3.0

    parser = argparse.ArgumentParser(
        description="Args for training parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, dest="output_dir", default=output_dir, help="Output directory")
    parser.add_argument(
        "--data-dir", type=str, default=data_dir, dest="data_dir", help="Directory containing the EYEPACS and APTOS datasets",
    )
    parser.add_argument("--use-validation", action="store_true", dest="use_validation", help="Whether to use validation set during training")
    parser.set_defaults(use_validation=True)
    parser.add_argument("--use-test", action="store_true", dest="use_test", help="Whether to use test set during training")
    parser.set_defaults(use_test=False)

    parser.add_argument(
        "--preproc-builder-config", type=str, default=preproc_builder_config, dest="preproc_builder_config", help="Builder config to store data in",
    )
    parser.add_argument(
        "--dr-decision-threshold", type=str, default=dr_decision_threshold, dest="dr_decision_threshold", help="Decision threshold for severity shift task",
    )
    
    parser.add_argument("--load-from-checkpoint", action="store_true", dest="load_from_checkpoint", help="Load from checkpoint?")
    parser.set_defaults(load_from_checkpoint=False)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=checkpoint_dir, dest="checkpoint_dir", help="Path where checkpoints are stored",
    )
    parser.add_argument("--cache-eval-datasets", action="store_false", dest="cache_eval_datasets", help="Cache evaluation datasets?")
    parser.set_defaults(cache_eval_datasets=False)
    
    