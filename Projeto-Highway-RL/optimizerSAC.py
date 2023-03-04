""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a OpenAI Gym environment.
This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.
"""
from typing import Any, Dict
from copy import deepcopy

import numpy as np
import gym
import highway_env
import optuna
from stable_baselines3 import SAC
from torch import nn as nn
from trialEvalCallback import TrialEvalCallback


ENV_ID = "racetrack-v0"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995])
    learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # Removed due to analysis made in the notebook
    # buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    train_freq = trial.suggest_categorical("train_freq", [16, 32, 64, 128, 256])
    tau = trial.suggest_categorical("tau", [0.001, 0.01, 0.05])
    gradient_steps = train_freq
    ent_coef = "auto"
    # Removed due to analysis made in the notebook
    #net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    #net_arch = {"small": [64, 64],"medium": [256, 256],"big": [400, 300]}[net_arch]

    target_entropy = "auto"

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
    }

    return hyperparams


def objectiveSAC(
    trial: optuna.Trial, n_evaluations: int = 5, n_timesteps: int | float = 1e4, n_eval_episodes: int = 20
    ) -> float:

    eval_freq = int(n_timesteps)/n_evaluations
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_sac_params(trial))
    # Create the RL model
    model = SAC(**kwargs)
    # Create env used for evaluation
    eval_env = gym.make(ENV_ID)
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq
    )

    nan_encountered = False
    try:
        model.learn(n_timesteps, callback=eval_callback)
    except (AssertionError, ValueError) as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward