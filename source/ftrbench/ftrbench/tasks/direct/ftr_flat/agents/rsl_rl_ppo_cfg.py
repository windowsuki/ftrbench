# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FtrFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "ftr_flat_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
class FtrRoughPPORunnerCfg(FtrFlatPPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 100
    experiment_name = "ftr_rough_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )


@configclass
class FtrReferenceRoughPPORunnerCfg(FtrFlatPPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 100
    experiment_name = "ftr_reference_rough_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


@configclass
class FtrReferenceTerrainPPORunnerCfg(FtrReferenceRoughPPORunnerCfg):
    max_iterations = 3000
    save_interval = 100
    experiment_name = "ftr_reference_terrain_direct"
