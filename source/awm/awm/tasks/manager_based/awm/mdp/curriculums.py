# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_goal(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    goal_distance: float,
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain curriculum based on goal-reaching success.

    Robots that reached the goal move up to a harder terrain level.
    Robots that timed out move down to an easier terrain level.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[env_ids, :2]
    goal_xy = env.scene.terrain.env_origins[env_ids, :2] + torch.tensor(
        [goal_distance, 0.0], device=env.device
    )
    dist = torch.norm(goal_xy - root_xy, dim=-1)

    move_up = dist < goal_radius
    move_down = ~move_up

    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(move_up.float()).item()
