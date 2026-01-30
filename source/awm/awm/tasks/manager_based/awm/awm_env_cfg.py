# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp
from .terrains import ROUGH_TERRAINS_CFG

# ----------------------------------------------------------------------------
# Robot USD config
# ----------------------------------------------------------------------------

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_USD_PATH = os.path.join(_WORKSPACE_ROOT, "custom_assets", "awm.usd")

AWM_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "wheel_F_L": 0.0,
            "wheel_F_R": 0.0,
            "wheel_B_R": 0.0,
            "wheel_B_L": 0.0,
            "leg_F_L": -2.53,
            "leg_F_R": -2.53,
            "leg_B_L": -2.53,
            "leg_B_R": -2.53,
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"],
            effort_limit=20000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=20000.0,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_.*"],
            # effort_limit=5000.0,
            effort_limit_sim=15000.0,
            # velocity_limit=10.0,
            velocity_limit_sim=10.0,
            stiffness=5000.0,
            damping=50.0,
        ),
    },
)


# ----------------------------------------------------------------------------
# Scene definition
# ----------------------------------------------------------------------------


@configclass
class AwmSceneCfg(InteractiveSceneCfg):
    """Configuration for the AWM scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}"
                "/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = AWM_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner_fl = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/awm/leg",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(1.0, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_fr = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/awm/leg_2",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(1.0, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_bl = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/awm/leg_3",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(1.0, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_br = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/awm/leg_4",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(1.0, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/awm/.*",
        history_length=3,
        track_air_time=True,
        track_pose=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ----------------------------------------------------------------------------
# MDP settings
# ----------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    drive = mdp.AwmDriveActionCfg(
        asset_name="robot",
        wheel_joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
        leg_joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
        max_wheel_speed=20.0,
        leg_offset=2.5,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        distance_to_goal = ObsTerm(func=mdp.distance_to_goal, params={"goal_distance": 2.0})
        base_lin_vel_x = ObsTerm(func=mdp.base_lin_vel_x)
        mean_wheel_speed = ObsTerm(
            func=mdp.mean_wheel_speed,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"]
                )
            },
        )
        mean_leg_pos = ObsTerm(
            func=mdp.mean_leg_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"]
                )
            },
        )
        leg_actions = ObsTerm(func=mdp.leg_actions, params={"num_wheels": 4})
        height_scan = ObsTerm(
            func=mdp.height_scan_multi,
            params={
                "sensor_names": [
                    "height_scanner_fl",
                    "height_scanner_fr",
                    "height_scanner_bl",
                    "height_scanner_br",
                ]
            },
        )
        wheel_contact_forces = ObsTerm(
            func=mdp.wheel_contact_forces,
            params={"sensor_name": "contact_forces", "body_names": "wheel_.*"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset events."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, -0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_legs = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.01)
    progress = RewTerm(func=mdp.progress_to_goal, weight=1.0, params={"goal_distance": 2.0})
    goal_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=10.0,
        params={"goal_distance": 2.0, "goal_radius": 0.5},
    )
    leg_extend_flat = RewTerm(
        func=mdp.leg_extension_penalty_flat,
        weight=-0.2,
        params={
            "sensor_names": [
                "height_scanner_fl",
                "height_scanner_fr",
                "height_scanner_bl",
                "height_scanner_br",
            ],
            "var_scale": 50.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"]),
        },
    )
    leg_extend_rough = RewTerm(
        func=mdp.leg_extension_reward_rough,
        weight=0.5,
        params={
            "sensor_names": [
                "height_scanner_fl",
                "height_scanner_fr",
                "height_scanner_bl",
                "height_scanner_br",
            ],
            "var_scale": 50.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"]),
        },
    )
    leg_extend_slip = RewTerm(
        func=mdp.leg_extension_reward_on_slip,
        weight=0.4,
        params={
            "wheel_radius": 0.1,
            "slip_threshold": 0.1,
            "min_speed": 0.05,
            "wheel_asset_cfg": SceneEntityCfg(
                "robot", joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"]
            ),
            "leg_asset_cfg": SceneEntityCfg(
                "robot", joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"]
            ),
        },
    )
    straightness_lat = RewTerm(func=mdp.lateral_velocity_penalty, weight=-0.05)
    straightness_yaw = RewTerm(func=mdp.yaw_rate_penalty, weight=-0.02)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"goal_distance": 2.0, "goal_radius": 0.5})


# ----------------------------------------------------------------------------
# Environment configuration
# ----------------------------------------------------------------------------


@configclass
class AwmEnvCfg(ManagerBasedRLEnvCfg):
    scene: AwmSceneCfg = AwmSceneCfg(num_envs=50, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    goal_distance: float = 2.0
    goal_radius: float = 0.5

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.observations.policy.distance_to_goal.params["goal_distance"] = self.goal_distance
        self.rewards.progress.params["goal_distance"] = self.goal_distance
        self.rewards.goal_bonus.params["goal_distance"] = self.goal_distance
        self.rewards.goal_bonus.params["goal_radius"] = self.goal_radius
        self.terminations.goal_reached.params["goal_distance"] = self.goal_distance
        self.terminations.goal_reached.params["goal_radius"] = self.goal_radius


@configclass
class AwmFlatEnvCfg(AwmEnvCfg):
    """Flat-terrain variant of the AWM environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
