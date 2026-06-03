# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FTR robot asset configuration for Isaac Lab 4.5."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import PhysxCfg, SimulationCfg

_ASSET_DIR = Path(__file__).resolve().parent


FTR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/pumbaa_wheel",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSET_DIR / "usd" / "ftr" / "ftr_v1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "baselink_wheel": ImplicitActuatorCfg(
            joint_names_expr=[
                *[f"L{i + 1}RevoluteJoint" for i in range(8)],
                *[f"R{i + 1}RevoluteJoint" for i in range(8)],
            ],
            stiffness=1.0,
            damping=100.0,
        ),
        "flipper_wheel": ImplicitActuatorCfg(
            joint_names_expr=[
                *[f"LF{i + 1}RevoluteJoint" for i in range(5)],
                *[f"LR{i + 1}RevoluteJoint" for i in range(5)],
                *[f"RL{i + 1}RevoluteJoint" for i in range(5)],
                *[f"RR{i + 1}RevoluteJoint" for i in range(5)],
            ],
            stiffness=1.0,
            damping=100.0,
        ),
        "flipper_joint": ImplicitActuatorCfg(
            joint_names_expr=[
                "front_left_flipper_joint",
                "front_right_flipper_joint",
                "rear_left_flipper_joint",
                "rear_right_flipper_joint",
            ],
            stiffness=3.0e4,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=180.0,
            armature=100.0,
        ),
    },
)


FTR_SIM_CFG = SimulationCfg(
    dt=1 / 120,
    render_interval=4,
    physx=PhysxCfg(
        min_position_iteration_count=32,
        max_velocity_iteration_count=0,
    ),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=5.0,
        dynamic_friction=5.0,
        restitution=0.0,
    ),
)
