# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "awm_morph"
    empirical_normalization = True
    logger = "wandb"
    wandb_project = "awm_morph"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPOWheelsOnlyRunnerCfg(PPORunnerCfg):
    """Ablation: wheels-only baseline, same architecture as adaptive policy."""
    experiment_name = "awm_wheels_only"
    wandb_project = "awm_morph"


@configclass
class PPOLegsOpenRunnerCfg(PPORunnerCfg):
    """Ablation: legs-fully-open baseline, same architecture as adaptive policy."""
    experiment_name = "awm_legs_open"
    wandb_project = "awm_morph"


@configclass
class PPOFlatRunnerCfg(PPORunnerCfg):
    experiment_name = "awm_manager_flat"
    max_iterations = 10000
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )