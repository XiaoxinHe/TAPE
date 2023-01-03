import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'pubmed'
    # Additional num of worker for data loading
    cfg.num_workers = 8
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = 0
    # Custom log file name
    cfg.logfile = None

    cfg.train = CN()
    # Learning rate for language model
    cfg.train.lr_lm = 1e-5
    # Learning rate for gnn model
    cfg.train.lr_gnn = 1e-3
    # Learning rate patience
    cfg.train.lr_patience = 20

    cfg.train.lr_decay = 0.5
    # inner epoch
    cfg.train.epochs = 200
    # outer epoch
    cfg.train.stages = 5
    # Dropout
    cfg.train.dropout = 0.0
    # runs
    cfg.train.runs = 4

    cfg.model = CN()
    cfg.model.gnn_type = 'GCNConv'
    cfg.model.gnn_nlayer = 4
    cfg.model.res = True

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
