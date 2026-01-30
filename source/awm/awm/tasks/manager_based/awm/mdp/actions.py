# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Maps normalized actions to per-wheel velocity and per-leg position targets.
class AwmDriveAction(ActionTerm):
    """Action term mapping per-wheel and per-leg inputs to actuator targets."""

    cfg: "AwmDriveActionCfg"

    def __init__(self, cfg: "AwmDriveActionCfg", env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self._wheel_ids, self._wheel_names = self.robot.find_joints(cfg.wheel_joint_names, preserve_order=True)
        self._leg_ids, self._leg_names = self.robot.find_joints(cfg.leg_joint_names, preserve_order=True)

        if len(self._wheel_ids) == 0:
            raise ValueError("No wheel joints matched the provided names.")
        if len(self._leg_ids) == 0:
            raise ValueError("No leg joints matched the provided names.")

        self._action_dim = len(self._wheel_ids) + len(self._leg_ids) #Action dimension is both for wheels and legs now which is going to be 8
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device) #Wheel targets
        self._wheel_targets = torch.zeros((self.num_envs, len(self._wheel_ids)), device=self.device)
        self._leg_targets = torch.zeros((self.num_envs, len(self._leg_ids)), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)

        wheel_cmd = self._raw_actions[:, : len(self._wheel_ids)]
        leg_cmd = self._raw_actions[:, len(self._wheel_ids) :]

        self._wheel_targets = wheel_cmd * self.cfg.max_wheel_speed

        default_leg_pos = self.robot.data.default_joint_pos[:, self._leg_ids]
        leg_offset = leg_cmd * self.cfg.leg_offset
        self._leg_targets = default_leg_pos + leg_offset

    def apply_actions(self):
        self.robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)
        self.robot.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids)


@configclass
# Holds joint names and scales for the AWM drive action.
class AwmDriveActionCfg(ActionTermCfg):
    """Configuration for the AWM drive action term."""

    class_type: type[ActionTerm] = AwmDriveAction
    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING
    leg_joint_names: list[str] = MISSING
    max_wheel_speed: float = 20.0
    leg_offset: float = 0.05
    debug_vis: bool = False
