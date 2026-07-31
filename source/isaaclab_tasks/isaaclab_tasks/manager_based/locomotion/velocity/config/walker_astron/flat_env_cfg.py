# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Walker Astron 双足人形机器人平地步态强化学习环境物理配置文件。
此文件继承了 Isaac Lab 的通用速度跟踪运动控制基类，并针对 Walker Astron 进行了：
1. 地形设置（平面）。
2. 机器人 USD 资产绑定与初始微蹲姿态设置。
3. 限制控制关节为 23 个核心自由度。
4. 注入步态周期相位时钟观测。
5. 对齐 G1 Flat 奖惩体系。
"""

import torch
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets import WALKER_ASTRON_CFG
from isaaclab_physx.physics import PhysxCfg

# === 核心配置：定义 23 个控制自由度（Active Joints） ===
ACTIVE_JOINTS = [
    "L_hip_pitch_joint",
    "R_hip_pitch_joint",
    "waist_yaw_joint",
    "L_hip_roll_joint",
    "R_hip_roll_joint",
    "waist_pitch_joint",
    "L_hip_yaw_joint",
    "R_hip_yaw_joint",
    "waist_roll_joint",
    "L_knee_pitch_joint",
    "R_knee_pitch_joint",
    "L_ankle_pitch_joint",
    "R_ankle_pitch_joint",
    "L_ankle_roll_joint",
    "R_ankle_roll_joint",
    "L_shoulder_pitch_joint",
    "R_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "R_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "R_shoulder_yaw_joint",
    "L_elbow_pitch_joint",
    "R_elbow_pitch_joint",
]


# === 自定义步态时钟观测函数 (6 维: sin_L, sin_R, cos_L, cos_R, ratio_L, ratio_R) ===
def gait_clock_obs(env) -> torch.Tensor:
    if not hasattr(env, "_gait_cycle"):
        env._gait_cycle = 1.0
        env._phase_offset = torch.tensor([0.4, 0.9], device=env.device)
        env._phase_ratio = torch.tensor([0.4, 0.4], device=env.device)

    # 使用环境内置的步长缓存作为时间轴，对齐 deployment 算法中的周期逻辑
    steps = env.episode_length_buf.float()
    t = steps * env.step_dt / env._gait_cycle

    # 左右对称相角（相差半个周期 0.5）
    phase_L = (t + env._phase_offset[0]) % 1.0
    phase_R = (t + env._phase_offset[1]) % 1.0

    sin_L = torch.sin(2 * torch.pi * phase_L)
    cos_L = torch.cos(2 * torch.pi * phase_L)
    sin_R = torch.sin(2 * torch.pi * phase_R)
    cos_R = torch.cos(2 * torch.pi * phase_R)

    ratio = env._phase_ratio.repeat(env.num_envs, 1)
    obs = torch.stack([sin_L, sin_R, cos_L, cos_R], dim=-1)
    return torch.cat([obs, ratio], dim=-1)


@configclass
class PhysicsCfg(PresetCfg):
    """物理刚体解算器参数配置。"""
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = default


@configclass
class WalkerAstronRewards(RewardsCfg):
    """步态训练的奖励函数设计。"""
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*_joint")}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_.*_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    # 头部属于锁定关节，无需动作偏离惩罚
    joint_deviation_head = None
    joint_vel_head = None


@configclass
class WalkerAstronFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """机器人任务总场景与运行参数配置类。"""
    rewards: WalkerAstronRewards = WalkerAstronRewards()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        super().__post_init__()
        
        # === 1. 地形配置 ===
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        
        # === 2. 机器人与环境数 ===
        self.scene.robot = WALKER_ASTRON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096
        
        # === 3. 躯干物理事件与终止条件 ===
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ".*torso_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link", ".*waist_.*", "torso_link"]
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        
        # === 4. 指令速度范围限制 ===
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        # === 5. 平衡与控制稳定性惩罚项 ===
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7

        # === 6. 精简动作与观测空间（限制为 23 个 Active Joints） ===
        self.actions.joint_pos.joint_names = ACTIVE_JOINTS
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)

        # === 7. 注入 6 维步态周期时钟观测项 ===
        self.observations.policy.gait_clock = ObsTerm(func=gait_clock_obs)
