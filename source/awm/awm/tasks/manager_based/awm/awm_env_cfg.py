# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp
from .terrains import ROUGH_TERRAINS_CFG

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_USD_PATH = os.path.join(_WORKSPACE_ROOT, "custom_assets", "awm.usd")


def _to_sim_gains(real_kp: float, real_kd: float, kp_ratio: float, kd_ratio: float) -> tuple[float, float]:
    """Map real-world gain values to simulation gains."""
    return real_kp / kp_ratio, real_kd / kd_ratio


_GAIN_RATIOS = dict(kp_ratio=150.0, kd_ratio=16.0)
_WHEEL_GAINS_REAL = dict(kp_real=0.0, kd_real=2.928)
_LEG_GAINS_REAL = dict(kp_real=1500.0, kd_real=5.456)

_WHEEL_KP_SIM, _WHEEL_KD_SIM = _to_sim_gains(
    _WHEEL_GAINS_REAL["kp_real"],
    _WHEEL_GAINS_REAL["kd_real"],
    _GAIN_RATIOS["kp_ratio"],
    _GAIN_RATIOS["kd_ratio"],
)
_LEG_KP_SIM, _LEG_KD_SIM = _to_sim_gains(
    _LEG_GAINS_REAL["kp_real"],
    _LEG_GAINS_REAL["kd_real"],
    _GAIN_RATIOS["kp_ratio"],
    _GAIN_RATIOS["kd_ratio"],
)

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
            # "leg_F_L": 0.0,
            # "leg_F_R": 0.0,
            # "leg_B_L": 0.0,
            # "leg_B_R": 0.0,
            "leg_F_L": -2.53,
            "leg_F_R": -2.53,
            "leg_B_L": -2.53,
            "leg_B_R": -2.53,
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"],
            effort_limit_sim=1.94,
            velocity_limit_sim=7.6,
            stiffness=_WHEEL_KP_SIM,
            damping=_WHEEL_KD_SIM,
            armature=0.0022,
            friction=0.038,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_.*"],
            effort_limit_sim=0.68,
            velocity_limit_sim=6.52,
            stiffness=_LEG_KP_SIM,
            damping=_LEG_KD_SIM,
            armature=0.0048,
            friction=0.006,
        ),
    },
)


@configclass
class AwmSceneCfg(InteractiveSceneCfg):
    """Configuration for the AWM scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=2.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}"
                "/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = AWM_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    drive = mdp.AwmDriveActionCfg(
        asset_name="robot",
        wheel_joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
        leg_joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
        max_wheel_speed=8.0,
        leg_offset=0.5,
        use_auto_extension=False,
        closed_at_upper_limit=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        distance_to_goal = ObsTerm(func=mdp.distance_to_goal, params={"goal_distance": 5.0})
        base_lin_vel_x = ObsTerm(func=mdp.base_lin_vel_x)
        wheel_velocities = ObsTerm(
            func=mdp.wheel_velocities,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
                )
            },
        )
        leg_positions = ObsTerm(
            func=mdp.leg_positions,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
                )
            },
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        leg_actions = ObsTerm(func=mdp.leg_actions, params={"num_wheels": 4})
        progress_slip_hist = ObsTerm(
            func=mdp.progress_slip_history,
            params={
                "wheel_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
                ),
                "wheel_radius": 0.1,
                "ema_alpha": 0.1,
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
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_legs = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    progress_to_goal = RewTerm(func=mdp.progress_to_goal, weight=2.0, params={"goal_distance": 5.0})
    forward_vel = RewTerm(func=mdp.forward_velocity_reward, weight=1.0)
    wheel_slip = RewTerm(
        func=mdp.wheel_slip_penalty,
        weight=-0.4,
        params={
            "wheel_radius": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
            ),
        },
    )
    leg_extension_efficiency = RewTerm(
        func=mdp.leg_extension_efficiency,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "tilt_threshold": 0.15,
        },
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_.*", "leg_.*"])},
    )
    action = RewTerm(func=mdp.action_l2, weight=-0.0005)


@configclass
class CurriculumCfg:
    """Terrain curriculum based on goal-reaching success."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_goal,
        params={"goal_distance": 5.0, "goal_radius": 1.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"goal_distance": 5.0, "goal_radius": 1})
    high_velocity = DoneTerm(func=mdp.high_base_velocity)


@configclass
class AwmEnvCfg(ManagerBasedRLEnvCfg):
    scene: AwmSceneCfg = AwmSceneCfg(num_envs=50, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    goal_distance: float = 5.0
    goal_radius: float = 1.0

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 30.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.observations.policy.distance_to_goal.params["goal_distance"] = self.goal_distance
        self.rewards.progress_to_goal.params["goal_distance"] = self.goal_distance
        self.terminations.goal_reached.params["goal_distance"] = self.goal_distance
        self.terminations.goal_reached.params["goal_radius"] = self.goal_radius
        self.curriculum.terrain_levels.params["goal_distance"] = self.goal_distance
        self.curriculum.terrain_levels.params["goal_radius"] = self.goal_radius


@configclass
class AwmFlatEnvCfg(AwmEnvCfg):
    """Flat-terrain variant of the AWM environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None
        self.actions.drive.leg_offset = 0.0
        self.observations.policy.goal_heading = ObsTerm(
            func=mdp.goal_heading_error,
            params={"goal_distance": self.goal_distance},
        )
