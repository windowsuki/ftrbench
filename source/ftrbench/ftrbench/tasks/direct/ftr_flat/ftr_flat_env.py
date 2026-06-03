# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FTR flat-ground training environment for Isaac Lab 4.5."""

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, VecEnvObs
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from ftrbench.assets import FTR_CFG, FTR_SIM_CFG
from ftrbench.assets.terrain import ReferenceTerrain


FTR_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=2.0,
    num_rows=4,
    num_cols=5,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.05, 0.18),
            step_width=0.35,
            platform_width=1.2,
            border_width=0.3,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.05, 0.18),
            step_width=0.35,
            platform_width=1.2,
            border_width=0.3,
            holes=False,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.05, 0.30),
            platform_width=1.2,
            border_width=0.3,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.05, 0.30),
            platform_width=1.2,
            border_width=0.3,
        ),
    },
)


@configclass
class FtrFlatEnvCfg(DirectRLEnvCfg):
    """Configuration for a simple fixed-direction FTR locomotion task."""

    episode_length_s = 20.0
    decimation = 4
    action_space = 6
    observation_space = 22
    state_space = 0

    sim: SimulationCfg = FTR_SIM_CFG
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=6.0, replicate_physics=True)
    robot_cfg: ArticulationCfg = FTR_CFG

    max_linear_speed = 0.6
    max_yaw_speed = 1.2
    target_forward_velocity = 0.35
    tracking_sigma = 0.18
    flipper_step_deg = 4.0
    flipper_limit_deg = 60.0
    drive_lever_arm = 0.5
    base_wheel_radius = 0.10
    flipper_wheel_radius = 0.09
    min_base_height = 0.08
    max_tilt_cos = 0.5
    use_base_height_termination = True
    spawn_x_offset_range = (0.0, 0.0)
    spawn_y_offset_range = (0.0, 0.0)
    spawn_z_offset = 0.0
    policy_controls_drive = True
    use_reference_observations = False
    use_reference_rewards = False
    use_reference_terrain = False
    use_reference_births = False
    reference_terrain_name = "cur_steps_down"
    reference_terrain_prim_path = "/World/ground"
    forward_velocity_range = (0.35, 0.35)
    terrain_scan_offset = 0.5
    terrain_scan_clip = 1.0
    crossing_target_x = 1.4
    crossing_lateral_limit = 1.0
    crossing_success_reward = 20.0
    crossing_out_of_range_penalty = 5.0
    crossing_success_distance = 0.25
    crossing_path_longitudinal_margin = 0.2
    crossing_path_lateral_margin = 0.1
    stuck_velocity_threshold = 0.02
    stuck_penalty = 0.2
    body_clearance_weight = 1.0

    alive_reward_weight = 0.05
    velocity_tracking_weight = 1.5
    forward_velocity_weight = 0.25
    lateral_velocity_weight = 0.4
    yaw_velocity_weight = 0.08
    upright_weight = 0.8
    action_weight = 0.01
    action_rate_weight = 0.02
    termination_penalty = 5.0

    base_right_joint_names = [f"R{i + 1}RevoluteJoint" for i in range(8)]
    base_left_joint_names = [f"L{i + 1}RevoluteJoint" for i in range(8)]
    flipper_joint_names = [
        "front_left_flipper_joint",
        "front_right_flipper_joint",
        "rear_left_flipper_joint",
        "rear_right_flipper_joint",
    ]
    flipper_drive_right_joint_names = [
        *[f"RL{i + 1}RevoluteJoint" for i in range(5)],
        *[f"RR{i + 1}RevoluteJoint" for i in range(5)],
    ]
    flipper_drive_left_joint_names = [
        *[f"LF{i + 1}RevoluteJoint" for i in range(5)],
        *[f"LR{i + 1}RevoluteJoint" for i in range(5)],
    ]

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.viewer.eye = (8.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class FtrRoughEnvCfg(FtrFlatEnvCfg):
    """Configuration for mixed slope/stair FTR locomotion."""

    episode_length_s = 25.0
    target_forward_velocity = 0.30
    tracking_sigma = 0.22
    max_linear_speed = 0.5
    max_yaw_speed = 1.0
    min_base_height = 0.06
    max_tilt_cos = 0.35
    use_base_height_termination = False
    spawn_x_offset_range = (-1.0, -0.6)
    spawn_y_offset_range = (-0.25, 0.25)
    spawn_z_offset = 0.15

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FTR_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        visual_material=None,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=6.0, replicate_physics=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (10.0, -10.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)


@configclass
class FtrReferenceRoughEnvCfg(FtrRoughEnvCfg):
    """Reference-inspired crossing task: fixed drive command, learned flippers, height-map observation."""

    episode_length_s = 30.0
    action_space = 4
    observation_space = 115

    target_forward_velocity = 0.25
    forward_velocity_range = (0.15, 0.30)
    policy_controls_drive = False
    use_reference_observations = True
    use_reference_rewards = True

    spawn_x_offset_range = (-1.4, -1.0)
    spawn_y_offset_range = (-0.2, 0.2)
    spawn_z_offset = 0.15
    crossing_target_x = 1.4
    crossing_lateral_limit = 1.25
    max_tilt_cos = 0.25

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/pumbaa_wheel/pumbaa_wheel/chassis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.55, 0.0, 2.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(2.10, 0.90), ordering="xy"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class FtrReferenceTerrainEnvCfg(FtrReferenceRoughEnvCfg):
    """Reference crossing task with migrated USD terrains and birth start/target points."""

    reference_terrain_name = "cur_steps_down"
    use_reference_terrain = True
    use_reference_births = True

    forward_velocity_range = (0.20, 0.30)
    crossing_success_reward = 100.0
    crossing_out_of_range_penalty = 5.0
    termination_penalty = 50.0
    max_tilt_cos = 0.18

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=0.0, replicate_physics=True)


class FtrFlatEnv(DirectRLEnv):
    """A flat-ground task that rewards the FTR for driving in the world +X direction."""

    cfg: FtrFlatEnvCfg | FtrRoughEnvCfg

    def __init__(self, cfg: FtrFlatEnvCfg | FtrRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_buffers()
        self._cache_joint_ids()

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        if self.cfg.use_reference_terrain:
            self._reference_terrain = ReferenceTerrain(
                self.cfg.reference_terrain_name,
                prim_path=self.cfg.reference_terrain_prim_path,
            )
            self._reference_terrain.spawn(self.scene.stage)
        else:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        if self.cfg.use_reference_observations:
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.use_reference_terrain:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.reference_terrain_prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._ensure_buffers()
        self._previous_actions[:] = self._actions
        self._actions[:] = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self._cache_joint_ids()

        if self.cfg.policy_controls_drive:
            self._commands[:, 0] = self.cfg.max_linear_speed * self._actions[:, 0]
            self._commands[:, 1] = self.cfg.max_yaw_speed * self._actions[:, 1]
            flipper_actions = self._actions[:, 2:]
        else:
            self._commands[:, 0] = self._target_forward_velocity[:, 0]
            self._commands[:, 1] = 0.0
            flipper_actions = self._actions

        right_linear = (2.0 * self._commands[:, 0] + self._commands[:, 1] * self.cfg.drive_lever_arm) / 2.0
        left_linear = (2.0 * self._commands[:, 0] - self._commands[:, 1] * self.cfg.drive_lever_arm) / 2.0

        base_right_targets = -right_linear.unsqueeze(-1).repeat(1, len(self._base_right_joint_ids))
        base_left_targets = -left_linear.unsqueeze(-1).repeat(1, len(self._base_left_joint_ids))
        flipper_right_targets = right_linear.unsqueeze(-1).repeat(1, len(self._flipper_drive_right_joint_ids))
        flipper_left_targets = left_linear.unsqueeze(-1).repeat(1, len(self._flipper_drive_left_joint_ids))

        self.robot.set_joint_velocity_target(
            base_right_targets / self.cfg.base_wheel_radius,
            joint_ids=self._base_right_joint_ids,
        )
        self.robot.set_joint_velocity_target(
            base_left_targets / self.cfg.base_wheel_radius,
            joint_ids=self._base_left_joint_ids,
        )
        self.robot.set_joint_velocity_target(
            flipper_right_targets / self.cfg.flipper_wheel_radius,
            joint_ids=self._flipper_drive_right_joint_ids,
        )
        self.robot.set_joint_velocity_target(
            flipper_left_targets / self.cfg.flipper_wheel_radius,
            joint_ids=self._flipper_drive_left_joint_ids,
        )

        flipper_delta = math.radians(self.cfg.flipper_step_deg) * flipper_actions
        flipper_pos = self._get_flipper_positions()
        flipper_targets = torch.clamp(
            flipper_pos + flipper_delta,
            min=-math.radians(self.cfg.flipper_limit_deg),
            max=math.radians(self.cfg.flipper_limit_deg),
        )

        self.robot.set_joint_position_target(
            flipper_targets * self._flipper_sign,
            joint_ids=self._flipper_joint_ids,
        )

    def _get_observations(self) -> VecEnvObs:
        self._ensure_buffers()
        if self.cfg.use_reference_observations:
            obs = torch.cat(
                (
                    self._get_height_scan_observation(),
                    self.robot.data.projected_gravity_b[:, :2],
                    self._target_forward_velocity,
                    self.robot.data.root_ang_vel_b,
                    self._get_flipper_positions(),
                ),
                dim=-1,
            )
            return {"policy": obs}

        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._get_flipper_positions(),
                self._target_forward_velocity,
                self._commands,
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._ensure_buffers()
        if self.cfg.use_reference_rewards:
            return self._get_reference_rewards()

        lin_vel_w = self.robot.data.root_lin_vel_w
        ang_vel_b = self.robot.data.root_ang_vel_b
        gravity_b = self.robot.data.projected_gravity_b

        velocity_error = lin_vel_w[:, 0] - self.cfg.target_forward_velocity
        velocity_tracking = torch.exp(-(velocity_error**2) / self.cfg.tracking_sigma)
        upright_error = torch.sum(gravity_b[:, :2] ** 2, dim=-1)
        action_rate = torch.sum((self._actions - self._previous_actions) ** 2, dim=-1)

        reward = self.cfg.alive_reward_weight
        reward += self.cfg.velocity_tracking_weight * velocity_tracking
        reward += self.cfg.forward_velocity_weight * lin_vel_w[:, 0]
        reward -= self.cfg.lateral_velocity_weight * torch.abs(lin_vel_w[:, 1])
        reward -= self.cfg.yaw_velocity_weight * torch.abs(ang_vel_b[:, 2])
        reward -= self.cfg.upright_weight * upright_error
        reward -= self.cfg.action_weight * torch.sum(self._actions**2, dim=-1)
        reward -= self.cfg.action_rate_weight * action_rate

        fallen = self._get_fallen()
        reward -= self.cfg.termination_penalty * fallen.float()

        self.extras["log"] = {
            "Episode Reward/velocity_tracking": self.cfg.velocity_tracking_weight * velocity_tracking.mean(),
            "Episode Reward/forward_velocity": self.cfg.forward_velocity_weight * lin_vel_w[:, 0].mean(),
            "Episode Reward/upright_penalty": -self.cfg.upright_weight * upright_error.mean(),
            "Metrics/forward_velocity": lin_vel_w[:, 0].mean(),
            "Metrics/target_forward_velocity": self._target_forward_velocity.mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.use_reference_rewards:
            terminated = self._get_reference_terminated()
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            return terminated, time_out

        terminated = self._get_fallen()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        self._ensure_buffers()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        if self.cfg.use_reference_births:
            start_pos, start_quat, target_pos = self._sample_reference_births(len(env_ids))
            default_root_state[:, :3] = start_pos
            default_root_state[:, 3:7] = start_quat
            default_root_state[:, 7:] = 0.0
        else:
            default_root_state[:, :3] += self._get_env_origins()[env_ids]
            default_root_state[:, :3] += self._sample_spawn_offsets(len(env_ids))
            target_pos = self._get_env_origins()[env_ids].clone()
            target_pos[:, 0] += self.cfg.crossing_target_x

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._commands[env_ids] = 0.0
        self._episode_start_pos[env_ids] = default_root_state[:, :3]
        self._target_pos[env_ids] = target_pos
        self._update_target_direction(env_ids)

        if not self.cfg.policy_controls_drive:
            v_min, v_max = self.cfg.forward_velocity_range
            if v_min != v_max:
                self._target_forward_velocity[env_ids, 0] = torch.empty(
                    len(env_ids), device=self.device
                ).uniform_(v_min, v_max)
            else:
                self._target_forward_velocity[env_ids, 0] = v_min

    def _ensure_buffers(self) -> None:
        if not hasattr(self, "_flipper_sign"):
            self._flipper_sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=self.device)
        if not hasattr(self, "_actions"):
            self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        if not hasattr(self, "_previous_actions"):
            self._previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        if not hasattr(self, "_commands"):
            self._commands = torch.zeros((self.num_envs, 2), device=self.device)
        if not hasattr(self, "_target_forward_velocity"):
            self._target_forward_velocity = torch.full(
                (self.num_envs, 1), self.cfg.target_forward_velocity, device=self.device
            )
        if not hasattr(self, "_episode_start_pos"):
            self._episode_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        if not hasattr(self, "_target_pos"):
            self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        if not hasattr(self, "_target_direction"):
            self._target_direction = torch.zeros((self.num_envs, 3), device=self.device)
            self._target_direction[:, 0] = 1.0
        if not hasattr(self, "_target_distance"):
            self._target_distance = torch.full((self.num_envs,), self.cfg.crossing_target_x, device=self.device)

    def _cache_joint_ids(self) -> None:
        if hasattr(self, "_flipper_joint_ids"):
            return

        self._flipper_joint_ids = [int(self.robot.find_joints(name)[0][0]) for name in self.cfg.flipper_joint_names]
        self._base_right_joint_ids = [
            int(self.robot.find_joints(name)[0][0]) for name in self.cfg.base_right_joint_names
        ]
        self._base_left_joint_ids = [int(self.robot.find_joints(name)[0][0]) for name in self.cfg.base_left_joint_names]
        self._flipper_drive_right_joint_ids = [
            int(self.robot.find_joints(name)[0][0]) for name in self.cfg.flipper_drive_right_joint_names
        ]
        self._flipper_drive_left_joint_ids = [
            int(self.robot.find_joints(name)[0][0]) for name in self.cfg.flipper_drive_left_joint_names
        ]

    def _get_flipper_positions(self) -> torch.Tensor:
        self._ensure_buffers()
        self._cache_joint_ids()
        return self.robot.data.joint_pos[:, self._flipper_joint_ids] * self._flipper_sign

    def _get_env_origins(self) -> torch.Tensor:
        return self._terrain.env_origins if hasattr(self, "_terrain") else self.scene.env_origins

    def _sample_reference_births(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._prepare_reference_birth_tensors()
        indices = (self._reference_birth_cursor + torch.arange(num_resets, device=self.device)) % len(
            self._reference_birth_start_pos
        )
        self._reference_birth_cursor = int((self._reference_birth_cursor + num_resets) % len(self._reference_birth_start_pos))
        return (
            self._reference_birth_start_pos[indices].clone(),
            self._reference_birth_start_quat[indices].clone(),
            self._reference_birth_target_pos[indices].clone(),
        )

    def _prepare_reference_birth_tensors(self) -> None:
        if hasattr(self, "_reference_birth_start_pos"):
            return
        if not hasattr(self, "_reference_terrain"):
            raise RuntimeError("Reference birth reset requested, but no reference terrain was loaded.")

        start_positions = []
        target_positions = []
        euler_orientations = []
        quat_orientations = []
        quat_indices = []
        euler_indices = []
        for index, info in enumerate(self._reference_terrain.births):
            start_positions.append(info["start_point"])
            target_positions.append(info["target_point"])
            orient = info["start_orient"]
            if len(orient) == 4:
                quat_indices.append(index)
                quat_orientations.append(orient)
            elif len(orient) == 3:
                euler_indices.append(index)
                euler_orientations.append(orient)
            else:
                raise ValueError(f"start_orient must have 3 Euler values or 4 quaternion values. Got: {orient}")

        self._reference_birth_start_pos = torch.tensor(start_positions, dtype=torch.float32, device=self.device)
        self._reference_birth_target_pos = torch.tensor(target_positions, dtype=torch.float32, device=self.device)
        self._reference_birth_start_quat = torch.zeros((len(start_positions), 4), dtype=torch.float32, device=self.device)

        if quat_indices:
            self._reference_birth_start_quat[quat_indices] = torch.tensor(
                quat_orientations, dtype=torch.float32, device=self.device
            )
        if euler_indices:
            eulers = torch.tensor(euler_orientations, dtype=torch.float32, device=self.device)
            self._reference_birth_start_quat[euler_indices] = quat_from_euler_xyz(
                eulers[:, 0], eulers[:, 1], eulers[:, 2]
            )
        self._reference_birth_cursor = 0

    def _update_target_direction(self, env_ids: Sequence[int]) -> None:
        path = self._target_pos[env_ids] - self._episode_start_pos[env_ids]
        path[:, 2] = 0.0
        distance = torch.norm(path[:, :2], dim=-1).clamp_min(1.0e-6)
        self._target_distance[env_ids] = distance
        self._target_direction[env_ids] = path / distance.unsqueeze(-1)

    def _sample_spawn_offsets(self, num_resets: int) -> torch.Tensor:
        offsets = torch.zeros((num_resets, 3), device=self.device)
        x_min, x_max = self.cfg.spawn_x_offset_range
        y_min, y_max = self.cfg.spawn_y_offset_range
        if x_min != x_max:
            offsets[:, 0] = torch.empty(num_resets, device=self.device).uniform_(x_min, x_max)
        else:
            offsets[:, 0] = x_min
        if y_min != y_max:
            offsets[:, 1] = torch.empty(num_resets, device=self.device).uniform_(y_min, y_max)
        else:
            offsets[:, 1] = y_min
        offsets[:, 2] = self.cfg.spawn_z_offset
        return offsets

    def _get_fallen(self) -> torch.Tensor:
        gravity_b = self.robot.data.projected_gravity_b
        fallen = gravity_b[:, 2] > -self.cfg.max_tilt_cos
        if self.cfg.use_base_height_termination:
            base_height = self.robot.data.root_pos_w[:, 2] - self._get_env_origins()[:, 2]
            fallen |= base_height < self.cfg.min_base_height
        return fallen

    def _get_height_scan_observation(self) -> torch.Tensor:
        ray_hits_z = self._height_scanner.data.ray_hits_w[..., 2]
        ray_hits_z = torch.nan_to_num(ray_hits_z, nan=0.0, posinf=0.0, neginf=0.0)
        scan = ray_hits_z - ray_hits_z.mean(dim=-1, keepdim=True)
        return scan.clamp(-self.cfg.terrain_scan_clip, self.cfg.terrain_scan_clip)

    def _get_reference_rewards(self) -> torch.Tensor:
        lin_vel_w = self.robot.data.root_lin_vel_w
        ang_vel_b = self.robot.data.root_ang_vel_b
        gravity_b = self.robot.data.projected_gravity_b
        root_pos = self.robot.data.root_pos_w

        progress_velocity = self._get_progress_velocity()
        forward_error = progress_velocity - self._target_forward_velocity[:, 0]
        velocity_tracking = torch.exp(-(forward_error**2) / self.cfg.tracking_sigma)
        orientation_penalty = torch.sum(gravity_b[:, :2] ** 2, dim=-1)
        angular_penalty = 0.1 * ang_vel_b[:, 0] ** 2 + 0.2 * ang_vel_b[:, 1] ** 2 + 0.1 * ang_vel_b[:, 2] ** 2
        stuck_penalty = (progress_velocity < self.cfg.stuck_velocity_threshold).float() * self.cfg.stuck_penalty

        body_clearance = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "_height_scanner"):
            ray_hits_z = torch.nan_to_num(self._height_scanner.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
            terrain_max = ray_hits_z.max(dim=-1).values
            body_clearance = torch.relu(root_pos[:, 2] - self.cfg.base_wheel_radius - terrain_max)

        success = self._get_reference_success()
        out_of_range = self._get_reference_out_of_range()
        fallen = self._get_fallen()

        reward = self.cfg.alive_reward_weight
        reward += self.cfg.velocity_tracking_weight * velocity_tracking
        reward -= self.cfg.upright_weight * orientation_penalty
        reward -= angular_penalty
        reward -= stuck_penalty
        reward -= self.cfg.body_clearance_weight * body_clearance
        reward -= self.cfg.action_rate_weight * torch.sum((self._actions - self._previous_actions) ** 2, dim=-1)
        reward += self.cfg.crossing_success_reward * success.float()
        reward -= self.cfg.crossing_out_of_range_penalty * out_of_range.float()
        reward -= self.cfg.termination_penalty * fallen.float()

        self.extras["log"] = {
            "Episode Reward/velocity_tracking": self.cfg.velocity_tracking_weight * velocity_tracking.mean(),
            "Episode Reward/upright_penalty": -self.cfg.upright_weight * orientation_penalty.mean(),
            "Episode Reward/body_clearance": -self.cfg.body_clearance_weight * body_clearance.mean(),
            "Metrics/forward_velocity": progress_velocity.mean(),
            "Metrics/target_forward_velocity": self._target_forward_velocity.mean(),
            "Metrics/crossing_success": success.float().mean(),
            "Metrics/distance_to_target": torch.norm(root_pos[:, :2] - self._target_pos[:, :2], dim=-1).mean(),
        }
        return reward

    def _get_progress_velocity(self) -> torch.Tensor:
        return torch.sum(self.robot.data.root_lin_vel_w * self._target_direction, dim=-1)

    def _get_reference_success(self) -> torch.Tensor:
        if self.cfg.use_reference_births:
            return torch.norm(self.robot.data.root_pos_w[:, :2] - self._target_pos[:, :2], dim=-1) <= (
                self.cfg.crossing_success_distance
            )
        return self.robot.data.root_pos_w[:, 0] >= self._target_pos[:, 0]

    def _get_reference_out_of_range(self) -> torch.Tensor:
        if self.cfg.use_reference_births:
            root_xy = self.robot.data.root_pos_w[:, :2]
            start_xy = self._episode_start_pos[:, :2]
            target_xy = self._target_pos[:, :2]
            center = 0.5 * (start_xy + target_xy)
            path = target_xy - start_xy
            distance = torch.norm(path, dim=-1).clamp_min(1.0e-6)
            direction = path / distance.unsqueeze(-1)
            normal = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
            relative = root_xy - center
            longitudinal = torch.sum(relative * direction, dim=-1)
            lateral = torch.sum(relative * normal, dim=-1)
            long_radius = 0.5 * distance + self.cfg.crossing_path_longitudinal_margin
            lat_radius = 0.25 * distance + self.cfg.crossing_path_lateral_margin
            return (longitudinal / long_radius) ** 2 + (lateral / lat_radius) ** 2 > 1.0

        root_pos = self.robot.data.root_pos_w
        env_origins = self._get_env_origins()
        behind_start = root_pos[:, 0] < self._episode_start_pos[:, 0] - 0.5
        past_target = root_pos[:, 0] > self._target_pos[:, 0] + 0.5
        lateral = torch.abs(root_pos[:, 1] - env_origins[:, 1]) > self.cfg.crossing_lateral_limit
        return behind_start | past_target | lateral

    def _get_reference_terminated(self) -> torch.Tensor:
        return self._get_fallen() | self._get_reference_success() | self._get_reference_out_of_range()
