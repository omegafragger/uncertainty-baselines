import argparse
import datetime
import os
import pathlib
import pprint
import time
import numpy as np

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

import torch
import tensorflow as tf
import torch_utils
import uncertainty_baselines as ub
import utils  # local file import
import wandb

from tensorboard.plugins.hparams import api as hp

# Other utilities
from torch_deterministic_args import training_args

# Models
from uncertainty_baselines.models.torch_models.resnet import resnet50
from uncertainty_baselines.models.torch_models.vgg import vgg16

DEFAULT_NUM_EPOCHS = 90

MODEL_DICT = {
    'resnet50': resnet50,
    'vgg16': vgg16
}


def main():
    train_args = training_args().parse_args()
    print ("Parsed args", train_args)

    # Output directory
    output_dir = os.path.join(train_args.output_dir, f'seed_{train_args.seed}')
    train_args.eyepacs_data_dir = os.path.join(train_args.data_dir, 'eyepacs')
    train_args.aptos_data_dir = os.path.join(train_args.data_dir, 'aptos')
    tf.io.gfile.makedirs(output_dir)
    logging.info('Saving checkpoints at %s', output_dir)

    # Set seeds
    tf.random.set_seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # Resolve CUDA devices
    if train_args.use_gpu and torch.cuda.is_available():
        print ('Running model with CUDA')
        device = 'cuda'
    else:
        print ('Running model with CPU')
        device = 'cpu'

    # Log Run Hypers
    hypers_dict = {
      'per_core_batch_size': train_args.per_core_batch_size,
      'base_learning_rate': train_args.base_learning_rate,
      'final_decay_factor': train_args.final_decay_factor,
      'one_minus_momentum': train_args.one_minus_momentum,
      'l2': train_args.l2
    }
    logging.info('Hypers:')
    logging.info(pprint.pformat(hypers_dict))

    # Initialize distribution strategy on flag-specified accelerator
    strategy = utils.init_distribution_strategy(force_use_cpu=False,
                                                use_gpu=True,
                                                tpu_name=None)

    # Reweighting loss for class imbalance
    class_reweight_mode = train_args.class_reweight_mode
    if class_reweight_mode == 'constant':
        class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
    else:
        class_weights = None


    # Load in datasets.
    per_core_batch_size = train_args.per_core_batch_size
    datasets, steps = utils.load_dataset(
        train_batch_size=per_core_batch_size,
        eval_batch_size=per_core_batch_size,
        flags=train_args,
        strategy=strategy)
    available_splits = list(datasets.keys())
    test_splits = [split for split in available_splits if 'test' in split]
    eval_splits = [
        split for split in available_splits
        if 'validation' in split or 'test' in split
    ]

    # Iterate eval datasets
    eval_datasets = {split: iter(datasets[split]) for split in eval_splits}
    dataset_train = datasets['train']
    train_steps_per_epoch = steps['train']

    summary_writer = tf.summary.create_file_writer(
        os.path.join(output_dir, 'summaries'))

    # Build model
    logging.info('Building Torch model')
    model = MODEL_DICT[train_args.model](spectral_normalization=train_args.sn,
                                         mod=train_args.mod,
                                         num_classes=1,
                                         coeff=train_args.coeff)
    logging.info(f'Model number of weights: {torch_utils.count_parameters(model)}')


    # Training hyper parameters
    base_lr = train_args.base_learning_rate
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=(1.0-train_args.one_minus_momentum),
        nesterov=True
    )
    steps_per_train_epoch = steps['train']
    steps_to_lr_peak = int(steps_per_train_epoch * train_args.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, steps_to_lr_peak, T_mult=2
    )
    model = model.to(device)

    # Define metrics
    metrics = utils.get_diabetic_retinopathy_base_metrics(
        use_tpu=False,
        num_bins = train_args.num_bins,
        use_validation = train_args.use_validation,
        available_splits = available_splits
    )

    metrics.update(
        utils.get_diabetic_retinopathy_cpu_metrics(
            available_splits=available_splits,
            use_validation=train_args.use_validation
        )
    )


    # Init loss function
    loss_fn = torch.nn.BCELoss()
    sigmoid = torch.nn.Sigmoid()
    max_steps = steps_per_train_epoch * train_args.train_epochs
    image_h = 512
    image_w = 512


    def run_train_epoch(iterator):

        def train_step(inputs):
            images = inputs['features']
            labels = inputs['labels']
            images = torch.from_numpy(images._numpy()).view(per_core_batch_size, 3,  # pylint: disable=protected-access
                                                            image_h,
                                                            image_w).to(device)
            labels = torch.from_numpy(labels._numpy()).to(device).float()  # pylint: disable=protected-access

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            logits = model(images)
            probs = sigmoid(logits).squeeze(-1)

            # Add L2 regularization loss to NLL
            negative_log_likelihood = loss_fn(probs, labels)
            l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = negative_log_likelihood + (FLAGS.l2 * l2_loss)

            # Backward/optimizer
            loss.backward()
            optimizer.step()

            # Convert to NumPy for metrics updates
            loss = loss.detach()
            negative_log_likelihood = negative_log_likelihood.detach()
            labels = labels.detach()
            probs = probs.detach()

            if device != 'cpu':
                loss = loss.cpu()
                negative_log_likelihood = negative_log_likelihood.cpu()
                labels = labels.cpu()
                probs = probs.cpu()

            loss = loss.numpy()
            negative_log_likelihood = negative_log_likelihood.numpy()
            labels = labels.numpy()
            probs = probs.numpy()

            metrics['train/loss'].update_state(loss)
            metrics['train/negative_log_likelihood'].update_state(
              negative_log_likelihood)
            metrics['train/accuracy'].update_state(labels, probs)
            metrics['train/auprc'].update_state(labels, probs)
            metrics['train/auroc'].update_state(labels, probs)
            metrics['train/ece'].add_batch(probs, label=labels)

        for step in tqdm(range(train_steps_per_epoch)):
            train_step(next(iterator))

            if step % 100 == 0:
                current_step = (epoch + 1) * step
                time_elapsed = time.time() - start_time
                steps_per_sec = float(current_step) / time_elapsed
                eta_seconds = (max_steps -
                               current_step) / steps_per_sec if steps_per_sec else 0
                message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                           'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                               current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                               steps_per_sec, eta_seconds / 60, time_elapsed / 60))
                logging.info(message)


    def run_eval_epoch(iterator, dataset_split, num_steps):

        def eval_step(inputs, model):
            images = inputs['features']
            labels = inputs['labels']
            images = torch.from_numpy(images._numpy()).view(per_core_batch_size, 3,  # pylint: disable=protected-access
                                                            image_h,
                                                            image_w).to(device)
            labels = torch.from_numpy(
              labels._numpy()).to(device).float().unsqueeze(-1)  # pylint: disable=protected-access

            with torch.no_grad():
                logits = torch.stack(
                    [model(images) for _ in range(FLAGS.num_dropout_samples_eval)],
                    dim=-1)

            # Logits dimension is (batch_size, 1, num_dropout_samples).
            logits = logits.squeeze()

            # It is now (batch_size, num_dropout_samples).
            probs = sigmoid(logits)

            # labels_tiled shape is (batch_size, num_dropout_samples).
            labels_tiled = torch.tile(labels, (1, FLAGS.num_dropout_samples_eval))

            log_likelihoods = -loss_fn(probs, labels_tiled)
            negative_log_likelihood = torch.mean(
              -torch.logsumexp(log_likelihoods, dim=-1) +
              torch.log(torch.tensor(float(FLAGS.num_dropout_samples_eval))))

            probs = torch.mean(probs, dim=-1)

            # Convert to NumPy for metrics updates
            negative_log_likelihood = negative_log_likelihood.detach()
            labels = labels.detach()
            probs = probs.detach()

            if device != 'cpu':
                negative_log_likelihood = negative_log_likelihood.cpu()
                labels = labels.cpu()
                probs = probs.cpu()

            negative_log_likelihood = negative_log_likelihood.numpy()
            labels = labels.numpy()
            probs = probs.numpy()

            metrics[dataset_split +
                  '/negative_log_likelihood'].update_state(negative_log_likelihood)
            metrics[dataset_split + '/accuracy'].update_state(labels, probs)
            metrics[dataset_split + '/auprc'].update_state(labels, probs)
            metrics[dataset_split + '/auroc'].update_state(labels, probs)
            metrics[dataset_split + '/ece'].add_batch(probs, label=labels)

        for _ in tqdm(range(num_steps)):
            eval_step(next(iterator), model=model)


    metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
    start_time = time.time()
    initial_epoch = 0
    train_iterator = iter(dataset_train)
    model.train()

    # Starting to train
    for epoch in range(initial_epoch, train_args.train_epochs):
        logging.info('Starting to run epoch: %s', epoch + 1)

        run_train_epoch(train_iterator)

        if train_args.use_validation:
            id_validation_iterator = iter(datasets['in_domain_validation'])
            logging.info('Starting to run in-domain validation eval at epoch: %s', epoch + 1)
            run_eval_epoch(id_validation_iterator, 'in_domain_validation',
                         steps['in_domain_validation'])

            ood_validation_iterator = iter(datasets['ood_validation'])
            logging.info('Starting to run OoD validation eval at epoch: %s', epoch + 1)
            run_eval_epoch(ood_validation_iterator, 'ood_validation',
                         steps['ood_validation'])

        id_test_iterator = iter(datasets['in_domain_test'])
        logging.info('Starting to run in-domain test eval at epoch: %s', epoch + 1)
        test_start_time = time.time()
        run_eval_epoch(id_test_iterator, 'in_domain_test', steps['in_domain_test'])
        ms_per_example = (time.time() - test_start_time) * 1e6 / per_core_batch_size
        metrics['in_domain_test/ms_per_example'].update_state(ms_per_example)

        ood_test_iterator = iter(datasets['ood_test'])
        logging.info('Starting to run OoD test eval at epoch: %s', epoch + 1)
        test_start_time = time.time()
        run_eval_epoch(ood_test_iterator, 'ood_test', steps['ood_test'])
        ms_per_example = (time.time() - test_start_time) * 1e6 / per_core_batch_size
        metrics['ood_test/ms_per_example'].update_state(ms_per_example)

        # Step scheduler
        scheduler.step()

        # Log and write to summary the epoch metrics
        utils.log_epoch_metrics(metrics=metrics, use_tpu=False)
        total_results = {name: metric.result() for name, metric in metrics.items()}
        # Metrics from Robustness Metrics (like ECE) will return a dict with a
        # single key/value, instead of a scalar.
        total_results = {
            k: (list(v.values())[0] if isinstance(v, dict) else v)
            for k, v in total_results.items()
        }
        with summary_writer.as_default():
          for name, result in total_results.items():
            tf.summary.scalar(name, result, step=epoch + 1)

        for metric in metrics.values():
          metric.reset_states()

        if (train_args.checkpoint_interval > 0 and
            (epoch + 1) % train_args.checkpoint_interval == 0):

          checkpoint_path = os.path.join(output_dir, f'model_{epoch + 1}.pt')
          torch_utils.checkpoint_torch_model(
              model=model,
              optimizer=optimizer,
              epoch=epoch + 1,
              checkpoint_path=checkpoint_path)
          logging.info('Saved Torch checkpoint to %s', checkpoint_path)

    final_checkpoint_path = os.path.join(output_dir,
                                       f'model_{train_args.train_epochs}.pt')
    torch_utils.checkpoint_torch_model(
      model=model,
      optimizer=optimizer,
      epoch=train_args.train_epochs,
      checkpoint_path=final_checkpoint_path)
    logging.info('Saved last checkpoint to %s', final_checkpoint_path)

    with summary_writer.as_default():
        hp.hparams({
            'base_learning_rate': train_args.base_learning_rate,
            'one_minus_momentum': train_args.one_minus_momentum,
            'l2': train_args.l2,
            'lr_warmup_epochs': train_args.lr_warmup_epochs
        })



if __name__ == '__main__':
    main()