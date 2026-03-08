# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Computes 2D (x-y plane) distance from the robot base to a fixed goal point
# located at +goal_distance along each environment's x-axis origin.
def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    dist = torch.norm(goal_xy - root_xy, dim=-1)
    return torch.nan_to_num(dist, nan=float(goal_distance), posinf=float(goal_distance))


# Dense progress reward term:
# keeps track of previous goal distance and rewards reduction in distance
# at each step (positive when moving toward the goal, negative otherwise).
class progress_to_goal(ManagerTermBase):
    """Reward for making progress toward a fixed goal in +x direction."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_dist = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        goal_distance = self.cfg.params["goal_distance"]
        dist = _goal_distance_xy(self._env, goal_distance, asset_cfg)
        self.prev_dist[env_ids] = dist[env_ids].clamp(max=20.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal_distance: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        dist = _goal_distance_xy(env, goal_distance, asset_cfg).clamp(max=20.0)
        progress = self.prev_dist - dist
        self.prev_dist[:] = dist
        return torch.nan_to_num(torch.clamp(progress, -1.0, 1.0), nan=0.0)

# Sparse success reward:
# returns 1.0 when robot is inside goal_radius from the target, else 0.0.
def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    goal_distance: float,
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dist = _goal_distance_xy(env, goal_distance, asset_cfg)
    return (dist < goal_radius).float()


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive forward velocity in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    return torch.nan_to_num(torch.clamp(vx, min=0.0, max=5.0), nan=0.0)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity (clamped to prevent physics-blowup spikes)."""
    asset: Articulation = env.scene[asset_cfg.name]
    val = torch.square(asset.data.root_lin_vel_b[:, 2])
    return torch.nan_to_num(torch.clamp(val, max=25.0), nan=0.0)


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity (clamped to prevent physics-blowup spikes)."""
    asset: Articulation = env.scene[asset_cfg.name]
    val = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.nan_to_num(torch.clamp(val, max=25.0), nan=0.0)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small smoothness penalty on joint speeds."""
    asset: Articulation = env.scene[asset_cfg.name]
    val = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return torch.nan_to_num(torch.clamp(val, max=100.0), nan=0.0)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small penalty on action magnitude."""
    return torch.mean(torch.square(env.action_manager.action), dim=1)


def leg_extension_efficiency(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize leg extension when terrain is easy (robot is level).

    On flat terrain the projected gravity vector stays close to (0, 0, -1), so
    the xy-component magnitude is near zero.  We scale the extension penalty by
    how easy the terrain currently is, creating a clear trade-off:
      - flat  → terrain_ease ≈ 1 → penalised for keeping legs out
      - rough → terrain_ease ≈ 0 → no extra penalty, legs can stay extended
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Leg extension normalised to [0, 1] (0 = closed/wheel, 1 = fully open/leg)
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]  # (N, L, 2)
    lower, upper = limits[..., 0], limits[..., 1]
    leg_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    extension = (leg_pos - lower) / (upper - lower + 1e-6)
    max_extension = torch.amax(torch.clamp(extension, 0.0, 1.0), dim=1)

    # Terrain difficulty from projected gravity xy deviation (0 = flat, >0 = tilted)
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    terrain_ease = 1.0 - torch.clamp(tilt / tilt_threshold, 0.0, 1.0)

    return max_extension * terrain_ease


def rough_terrain_speed_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize high forward speed when terrain is difficult.

    Teaches the robot to slow down on rough terrain rather than blasting
    forward at full wheel speed and getting launched.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    speed = torch.nan_to_num(torch.abs(asset.data.root_lin_vel_b[:, 0]), nan=0.0)
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    terrain_difficulty = torch.clamp(tilt / tilt_threshold, 0.0, 1.0)
    return torch.clamp(speed * terrain_difficulty, max=5.0)


def wheel_slip_penalty(
    env: ManagerBasedRLEnv,
    wheel_radius: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel slip: wheel linear speed exceeding base forward speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_omega = asset.data.joint_vel[:, asset_cfg.joint_ids]
    wheel_lin_speed = torch.abs(wheel_omega) * wheel_radius
    mean_wheel_speed = torch.mean(wheel_lin_speed, dim=1)
    base_vx = torch.abs(asset.data.root_lin_vel_b[:, 0])
    return torch.nan_to_num(torch.clamp(mean_wheel_speed - base_vx, min=0.0, max=5.0), nan=0.0)
