import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

import torch_utils  # local file import
import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# PyTorch crash with CUDA OoM error.
tf.config.experimental.set_visible_devices([], 'GPU')

DEFAULT_NUM_EPOCHS = 90


# Data load / output flags.

class FLAGS:
    output_dir = '/home/jishnu/Projects/db_retino_outputs'
    data_dir = '/home/jishnu/Projects/ub_retinopathy'
    use_validation = True
    base_learning_rate = 4e-4
    one_minus_momentum = 0.1
    lr_warmup_epochs = 20
    seed = 42
    l2 = 5e-5
    train_epochs = DEFAULT_NUM_EPOCHS
    train_batch_size = 16
    eval_batch_size = 32
    checkpoint_interval = 25
    dropout_rate = 0.1
    num_dropout_samples_eval = 10
    num_bins = 15
    use_gpu = True

tf.io.gfile.makedirs(FLAGS.output_dir)
logging.info('Saving checkpoints at %s', FLAGS.output_dir)

# Set seeds
tf.random.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)

# Resolve CUDA device(s)
if FLAGS.use_gpu and torch.cuda.is_available():
    print('Running model with CUDA.')
    device = 'cuda:0'
else:
    print('Running model on CPU.')
    device = 'cpu'

train_batch_size = FLAGS.train_batch_size
eval_batch_size = FLAGS.eval_batch_size // FLAGS.num_dropout_samples_eval


train_batch_size = FLAGS.train_batch_size
eval_batch_size = FLAGS.eval_batch_size // FLAGS.num_dropout_samples_eval

# As per the Kaggle challenge, we have split sizes:
# train: 35,126
# validation: 10,906
# test: 42,670
ds_info = tfds.builder('diabetic_retinopathy_detection').info
steps_per_epoch = ds_info.splits['train'].num_examples // train_batch_size
steps_per_validation_eval = (
  ds_info.splits['validation'].num_examples // eval_batch_size)
steps_per_test_eval = ds_info.splits['test'].num_examples // eval_batch_size

data_dir = FLAGS.data_dir

dataset_train_builder = ub.datasets.get('ub_diabetic_retinopathy_detection', split='train', data_dir=data_dir)
dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

dataset_validation_builder = ub.datasets.get(
  'ub_diabetic_retinopathy_detection',
  split='validation',
  data_dir=data_dir,
  is_training=not FLAGS.use_validation)
validation_batch_size = (
  eval_batch_size if FLAGS.use_validation else train_batch_size)
dataset_validation = dataset_validation_builder.load(
  batch_size=validation_batch_size)
if not FLAGS.use_validation:
# Note that this will not create any mixed batches of train and validation
# images.
    dataset_train = dataset_train.concatenate(dataset_validation)

dataset_test_builder = ub.datasets.get(
  'ub_diabetic_retinopathy_detection', split='test', data_dir=data_dir)
dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)


train_iterator = iter(dataset_train)
val_iterator = iter(dataset_validation)
test_iterator = iter(dataset_test)

# Check dataset stats

def get_torch_inputs(inputs, batch_size, image_h, image_w, device):
    print ('Here')
    images = inputs['features']
    labels = inputs['labels']
    print ('Here')
    images = torch.from_numpy(images._numpy()).view(batch_size, 3, image_h, image_w).to(device)
    labels = torch.from_numpy(labels._numpy()).to(device).float()
    return images, labels

# Train
for step in range(steps_per_epoch):
    inputs = next(train_iterator)
    images, labels = get_torch_inputs(inputs, train_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break
    
# Validation
for step in range(steps_per_epoch):
    inputs = next(val_iterator)
    images, labels = get_torch_inputs(inputs, eval_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break

# Test
for step in range(steps_per_epoch):
    inputs = next(test_iterator)
    images, labels = get_torch_inputs(inputs, eval_batch_size, image_h, image_w, device)
    print (images.shape)
    print (labels.shape)
    print (images.min())
    print (images.max())
    print (images.unique())
    print (labels.unique())
    break

