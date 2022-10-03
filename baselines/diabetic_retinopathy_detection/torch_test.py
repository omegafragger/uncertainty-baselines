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


class FLAGS:
    output_dir = '/scratch-ssd/jisoti/ub_retinopathy_outputs'
    eyepacs_data_dir = '/scratch-ssd/jisoti/ub_retinopathy/eyepacs'
    aptos_data_dir = '/scratch-ssd/jisoti/ub_retinopathy/aptos'
    use_validation = True
    use_test = True
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
    
    distribution_shift = 'severity'
    load_train_split = True
    
    base_learning_rate = 0.023072
    final_decay_factor = 0.01
    one_minus_momentum = 0.0098467
    lr_schedule = 'step'
    lr_warmup_epochs = 1
    lr_decay_ratio = 0.2
    lr_decay_epochs = ['30', '60']
    
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


tf.random.set_seed(FLAGS.seed)
wandb_run = None
output_dir = FLAGS.output_dir

tf.io.gfile.makedirs(output_dir)

# Log Run Hypers
hypers_dict = {
  'per_core_batch_size': FLAGS.per_core_batch_size,
  'base_learning_rate': FLAGS.base_learning_rate,
  'final_decay_factor': FLAGS.final_decay_factor,
  'one_minus_momentum': FLAGS.one_minus_momentum,
  'l2': FLAGS.l2
}

# Initialize distribution strategy on flag-specified accelerator
strategy = utils.init_distribution_strategy(FLAGS.force_use_cpu,
                                          FLAGS.use_gpu, FLAGS.tpu)
use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)
per_core_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores


class_reweight_mode = FLAGS.class_reweight_mode
if class_reweight_mode == 'constant':
    class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
else:
    class_weights = None


datasets, steps = utils.load_dataset(
  train_batch_size=per_core_batch_size,
  eval_batch_size=per_core_batch_size,
  flags=FLAGS,
  strategy=strategy)


dataset_iterators = {
    'train': iter(datasets['train']),
    'in_domain_validation': iter(datasets['in_domain_validation']),
    'ood_validation': iter(datasets['ood_validation']),
    'in_domain_test': iter(datasets['in_domain_test']),
    'ood_test': iter(datasets['ood_test'])
}

import torch

def get_torch_inputs(inputs, batch_size, image_h, image_w, device):
    images = inputs['features']
    labels = inputs['labels']
    images = torch.from_numpy(images._numpy()).view(batch_size, 3, image_h, image_w).to(device)
    labels = torch.from_numpy(labels._numpy()).to(device).float()
    return images, labels

# Train
image_h = 512
image_w = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# iD Train
for step in range(steps['train']):
    inputs = next(dataset_iterators['train'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    images = images.to(device)
    labels = labels.to(device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break
    
# iD val
for step in range(steps['in_domain_validation']):
    inputs = next(dataset_iterators['in_domain_validation'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break

# OoD Val
for step in range(steps['ood_validation']):
    inputs = next(dataset_iterators['ood_validation'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break
    
# iD test
for step in range(steps['in_domain_test']):
    inputs = next(dataset_iterators['in_domain_test'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break

# OoD test
for step in range(steps['ood_test']):
    inputs = next(dataset_iterators['ood_test'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break


from tqdm import tqdm

# iD Train
for step in tqdm(range(steps['train'])):
    inputs = next(dataset_iterators['train'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    images = images.to(device)
    labels = labels.to(device)
    pass
    
# iD val
for step in tqdm(range(steps['in_domain_validation'])):
    inputs = next(dataset_iterators['in_domain_validation'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    images = images.to(device)
    labels = labels.to(device)
    pass

# OoD Val
for step in tqdm(range(steps['ood_validation'])):
    inputs = next(dataset_iterators['ood_validation'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    images = images.to(device)
    labels = labels.to(device)
    pass
    
# iD test
for step in tqdm(range(steps['in_domain_test'])):
    inputs = next(dataset_iterators['in_domain_test'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    images = images.to(device)
    labels = labels.to(device)
    pass

# OoD test
for step in tqdm(range(steps['ood_test'])):
    inputs = next(dataset_iterators['ood_test'])
    images, labels = get_torch_inputs(inputs, per_core_batch_size, image_h, image_w, device)
    images = images.to(device)
    labels = labels.to(device)
    pass
