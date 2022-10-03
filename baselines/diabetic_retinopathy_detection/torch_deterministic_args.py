import argparse


DEFAULT_NUM_EPOCHS = 90

def training_args():
    '''
    Training args for pytorch script.
    For full details on training args, please have a look at: https://github.com/omegafragger/uncertainty-baselines/blob/main/baselines/diabetic_retinopathy_detection/deterministic.py
    '''
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
    per_core_batch_size = 1
    checkpoint_interval = 25
    num_bins = 15
    
    use_gpu = True
    num_cores = 1
    
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

    parser.add_argument("--cache-eval-datasets", action="store_false", dest="cache_eval_datasets", help="Cache evaluation datasets?")
    parser.set_defaults(cache_eval_datasets=False)

    
    parser.add_argument("--use-wandb", action="store_false", dest="use_wandb", help="Use Wandb?")
    parser.set_defaults(use_wandb=False)
    parser.add_argument("--wandb-dir", type=str, default=wandb_dir, dest="wandb_dir", help="Wandb Directory")
    parser.add_argument("--project", type=str, default=project, dest="project", help="Wandb Project")
    parser.add_argument("--exp-name", type=str, default=exp_name, dest="exp_name", help="Wandb Experiment Name")
    parser.add_argument("--exp-group", type=str, default=exp_group, dest="exp_group", help="Wandb Experiment Group")

    parser.add_argument("--distribution-shift", type=str, default=distribution_shift, dest="distribution_shift", help="Type of distribution shift setup: severity/aptos")
    parser.add_argument("--load-train-split", action="store_true", dest="load_train_split", help="Load train split?")
    parser.set_defaults(load_train_split=True)


    parser.add_argument("--base-learning-rate", type=float, default=base_learning_rate, dest="base_learning_rate", help="Base Learning rate")
    parser.add_argument("--final-decay-factor", type=float, default=final_decay_factor, dest="final_decay_factor", help="Final Decay factor")
    parser.add_argument("--one-minus-momentum", type=float, default=one_minus_momentum, dest="one_minus_momentum", help="One minus momentum")
    parser.add_argument("--lr-schedule", type=str, default=lr_schedule, dest="lr_schedule", help="LR Schedule")
    parser.add_argument("--lr-warmup-epochs", type=int, default=lr_warmup_epochs, dest="lr_warmup_epochs", help="LR Warmup Epochs")
    parser.add_argument("--lr-decay-ratio", type=float, default=lr_decay_ratio, dest="lr_decay_ratio", help="LR Decay Ratio")
    parser.add_argument("--lr-decay-epoch-1", type=int, default=lr_decay_epoch_1, dest="lr_decay_epoch_1", help="LR Decay Epoch 1")
    parser.add_argument("--lr-decay-epoch-2", type=int, default=lr_decay_epoch_2, dest="lr_decay_epoch_2", help="LR Decay Epoch 2")

    parser.add_argument("--seed", type=int, default=seed, dest="seed", help="Seed")
    parser.add_argument("--class-reweight-mode", type=str, default=class_reweight_mode, dest="class_reweight_mode", help="Mode for class weighting for unbalanced classes")
    parser.add_argument("--l2", type=float, default=l2, dest="l2", help="L2 regularisation coefficient")
    parser.add_argument("--train-epochs", type=int, default=train_epochs, dest="train_epochs", help="Number of training epochs")
    parser.add_argument("--per-core-batch-size", type=int, default=per_core_batch_size, dest="per_core_batch_size", help="Batch size per GPU")
    parser.add_argument("--checkpoint-interval", type=int, default=checkpoint_interval, dest="checkpoint_interval", help="Interval between two checkpoints")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins", help="Number of bins for ECE computation")

    parser.add_argument("--use-gpu", action="store_true", dest="use_gpu", help="Use GPU for training?")
    parser.set_defaults(use_gpu=True)
    parser.add_argument("--num-cores", type=int, default=num_cores, dest="num_cores", help="Number of cores for training (GPUs)")


    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")
    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)

    return parser