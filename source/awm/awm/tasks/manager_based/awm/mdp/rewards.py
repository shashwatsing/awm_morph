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


def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    return torch.norm(goal_xy - root_xy, dim=-1)


class progress_to_goal(ManagerTermBase):
    """Reward for making progress toward a fixed goal in +x direction."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_dist = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        goal_distance = self.cfg.params["goal_distance"]
        dist = _goal_distance_xy(self._env, goal_distance, asset_cfg)
        self.prev_dist[env_ids] = dist[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal_distance: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        dist = _goal_distance_xy(env, goal_distance, asset_cfg)
        progress = self.prev_dist - dist
        self.prev_dist[:] = dist
        return progress


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    goal_distance: float,
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dist = _goal_distance_xy(env, goal_distance, asset_cfg)
    return (dist < goal_radius).float()


def _terrain_flatness(env, sensor_names: str | list[str], var_scale: float) -> torch.Tensor:
    if isinstance(sensor_names, str):
        sensor_names = [sensor_names]
    vars = []
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        heights = ray_hits - sensor.data.pos_w[:, 2:3]
        heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
        vars.append(torch.var(heights, dim=1, unbiased=False))
    var = torch.stack(vars, dim=1).mean(dim=1)
    return torch.exp(-var * var_scale)


def _leg_extension_mean(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    # if not hasattr(env, "_debug_printed_leg_limits"):
    #     print("leg joint limits (rad):", limits[0].detach().cpu())
    #     env._debug_printed_leg_limits = True
    lower = limits[..., 0]
    upper = limits[..., 1]
    denom = torch.clamp(upper - lower, min=1.0e-6)
    extension = (pos - lower) / denom
    return torch.mean(torch.clamp(extension, 0.0, 1.0), dim=1)


def leg_extension_penalty_flat(
    env: ManagerBasedRLEnv,
    sensor_names: str | list[str],
    var_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    flatness = _terrain_flatness(env, sensor_names, var_scale)
    extension = _leg_extension_mean(env, asset_cfg)
    return flatness * extension


def leg_extension_reward_rough(
    env: ManagerBasedRLEnv,
    sensor_names: str | list[str],
    var_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    flatness = _terrain_flatness(env, sensor_names, var_scale)
    extension = _leg_extension_mean(env, asset_cfg)
    return (1.0 - flatness) * extension


def leg_extension_reward_on_slip(
    env: ManagerBasedRLEnv,
    wheel_radius: float,
    slip_threshold: float,
    min_speed: float,
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward leg extension when wheels spin without forward progress."""
    asset: Articulation = env.scene[wheel_asset_cfg.name]
    wheel_omega = asset.data.joint_vel[:, wheel_asset_cfg.joint_ids]
    wheel_lin = torch.abs(wheel_omega) * wheel_radius
    base_vx = torch.abs(asset.data.root_lin_vel_b[:, 0]).unsqueeze(-1)
    slip = torch.clamp(wheel_lin - base_vx, min=0.0)
    slip_level = torch.clamp((slip - slip_threshold) / max(slip_threshold, 1.0e-6), 0.0, 1.0)
    need_help = torch.where(base_vx > min_speed, torch.zeros_like(base_vx), torch.ones_like(base_vx))
    extension = _leg_extension_mean(env, leg_asset_cfg).unsqueeze(-1)
    return torch.mean(slip_level * need_help * extension, dim=1)

def lateral_velocity_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_lin_vel_b[:, 1])

def yaw_rate_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_ang_vel_b[:, 2])
