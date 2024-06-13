#!/usr/bin/env python3

import argparse
import os
import random
import paramiko
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import evaluate_agent


def main():

    client = connect()
    config_paths = ["/home/x/VLNCE/VLN-CE/vlnce_baselines/config/r2r_baselines/cma_pm_aug_tune.yaml"]  # Adjust the path to your config file
    opts = ["MODEL.DEPTH_ENCODER.output_size", "128"]  # Example of overriding config options if needed

    config = get_config(config_paths, opts)
    config.defrost()

    # Print out the relevant parts of the configuration
    print("Depth Encoder Config:", config.MODEL.DEPTH_ENCODER)
    print("ENV_NAME:", config.MODEL.INSTRUCTION_ENCODER)
    print("RGB Encoder Config:", config.MODEL.RGB_ENCODER)
    print("State Encoder Config:", config.MODEL.STATE_ENCODER)

    config.freeze()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type", default="eval",
        choices=["train", "eval", "inference"],
        required=False,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config", default="vlnce_baselines/config/r2r_baselines/cma_pm_da_aug_tune.yaml",
        type=str,
        required=False,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(client,**vars(args))
    client.close()


def run_exp(client, exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "eval":
        trainer.eval(client=client)

def connect():
        client = paramiko.SSHClient()
        # Initialize
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Set key policy
        client.connect('172.20.10.6', 22, 'mi', '123')
        # Connect to the host
        return client

if __name__ == "__main__":
    main()
