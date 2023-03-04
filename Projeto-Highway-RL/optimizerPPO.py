""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a OpenAI Gym environment.
This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.
"""
from typing import Any, Dict


import numpy as np
import gym
import highway_env
import optuna
from stable_baselines3 import PPO
from torch import nn as nn
from trialEvalCallback import TrialEvalCallback

ENV_ID = "racetrack-v0"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995])
    learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01])
    ent_coef = trial.suggest_categorical("ent_coef", [0.0001, 0.001, 0.01, 0.1])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    #gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    # Removed due to analysis made in the notebook
    #net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

    #net_arch = {"small": [dict(pi=[64, 64], vf=[64, 64])], "medium": [dict(pi=[256, 256], vf=[256, 256])],}[net_arch]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
    }

def objectivePPO(
    trial: optuna.Trial, n_evaluations: int = 5, n_timesteps: int | float = 5e4, n_eval_episodes: int = 20
    ) -> float:

    eval_freq = int(n_timesteps)/n_evaluations
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model
    model = PPO(**kwargs)
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
    except AssertionError as e:
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