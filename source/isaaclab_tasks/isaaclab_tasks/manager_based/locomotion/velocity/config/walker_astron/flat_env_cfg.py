# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Walker Astron 双足人形机器人平地步态强化学习环境物理配置文件。
此文件继承了 Isaac Lab 的通用速度跟踪运动控制基类，并针对 Walker Astron 进行了：
1. 地形设置（平面）。
2. 机器人 USD 资产绑定与初始微蹲姿态设置。
3. 奖励与惩罚项（Reward / Penalty Terms）的参数微调，引导机器人走出平稳健壮的步态。
"""

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets import WALKER_ASTRON_CFG
from isaaclab_physx.physics import PhysxCfg


@configclass
class PhysicsCfg(PresetCfg):
    """
    物理刚体解算器参数配置。
    """
    # 限制 GPU 缓存中的最大刚体面片数量，该数值决定了并发训练时系统能承载的最大刚体和接触点的上限。
    # 设置为 10 * 2^15，以防止在多环境并发时显卡显存溢出或解算器报错。
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = default


@configclass
class WalkerAstronRewards(RewardsCfg):
    """
    步态训练的奖励函数设计（MDP - 马尔可夫决策过程中的奖励机制）。
    强化学习算法将通过极大化这些打分项之和来学习行走。正值代表鼓励，负值代表惩罚。
    """
    
    # 1. 致命惩罚：如果机器人摔倒（base 与地面产生非法碰撞触地并触发 Episode 提前重置），扣除 200 分。
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # 2. 线速度跟踪奖励：奖励机器人实际的 xy 平面线速度贴近给定的目标线速度指令。
    # 指数奖励形式（exp）：越接近目标速度，得分以指数曲线上升，最高得 1.0 分。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    
    # 3. 角速度跟踪奖励：奖励机器人实际的 z 轴偏航（Yaw）角速度贴近给定的旋转命令（控制转弯和原地打转避障）。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # 4. 离地时间奖励 (Air Time)：非常关键的双足运动引导项。
    # 当检测到脚底传感器离地（受力为零）并维持在空中一定时间后给予正分。
    # 目的：强制机器人双脚交替抬起跨步，而不是像雪橇一样在地上贴地滑行。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            # 监测左右脚踝的碰撞链接名（.*ankle_roll_link）的接触力
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.2, # 抬脚持续时间阈值达到 0.4 秒以上开始给分
        },
    )
    
    # 5. 惩罚脚部横向滑动 (Feet Slipping)：
    # 当足底与地面接触并承重时，如果足底在水平方向发生了滑动，扣除分数。
    # 目的：防止机器人在起步和跨步时脚底打滑，增加着地时的稳定感。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    
    # 6. 惩罚踝关节触极限限制 (Ankle Joint Limits)：
    # 如果脚踝关节转动角度太夸张，触碰到了几何限位，给予负分。
    # 目的：避免脚踝弯曲到反人类的角度，保护电机。
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*_joint")}
    )
    
    # 7. 惩罚身体关节的无用摆动 (Joint Deviation)：
    # 在双足步行中，上肢手臂和髋关节的不必要左右扭动、前后疯狂晃动会引起重心失衡。
    # 这里当检测到关节偏离“默认参考姿态”时，给予小幅惩罚，迫使机器人保持上身端庄和髋部姿态端正。
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
    # 8. 惩罚腰部关节偏离默认挺直站姿（权重提高到 -2.5，极力压制腰部左右侧弯和扭动）
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    # 9. 惩罚头部关节偏离默认前方位置（权重提高到 -1.5，强制锁死脖子，让相机直视前方）
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["head_.*_joint"])},
    )
    # 10. 惩罚头部关节角速度（压制高频振荡晃动）
    joint_vel_head = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["head_.*_joint"])},
    )


@configclass
class WalkerAstronFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    机器人任务总场景与运行参数配置类。
    """
    rewards: WalkerAstronRewards = WalkerAstronRewards()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self):
        # 执行基类的后初始化，构建基础物理世界
        super().__post_init__()
        
        # === 1. 地形配置 ===
        # 将原默认的粗糙起伏地形改写为平坦地面 (plane)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 屏蔽激光雷达对地面高度差的扫描仪 (平坦地面无需感知地形高低)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        
        # === 2. 机器人与平行环境数配置 ===
        # 将场景中的默认仿真机器人替换为我们专属的 Walker Astron 物理刚体配置
        self.scene.robot = WALKER_ASTRON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 将并发环境数修改为 1024（默认为 4096，大模型复制 4096 份会吃满 128GB 内存导致系统极慢甚至爆内存）
        self.scene.num_envs = 4096
        
        # === 3. 重心、躯干物理事件与终止条件绑定 ===
        # 绑定质量随机扰动（Events）和终止碰撞条件（Base Contact）到 Astron 的胸腔 link 上
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ".*torso_link"
        # 绑定终止碰撞条件（Base Contact）到 Astron 的屁股(base_link)、腰部(waist)以及胸腔(torso_link)
        # 这样只要这几个核心部位中任意一个着地，都会触发瞬间刷新
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link", ".*waist_.*", "torso_link"]
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        
        # === 4. 指令速度范围限制 ===
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)  # 允许后退(-0.5m/s)到前进(1.0m/s)的期望速度
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # 禁用横向螃蟹步移动
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0) # 允许最大偏航转向速度 (-1.0 到 1.0 rad/s)
        
        # === 5. 平衡性能与控制频率平滑微调 ===
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0 # 极强惩罚躯干（Base）的歪斜，从 -3.0 提高到 -5.0，迫使机器人保持脊椎垂直
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005    # 惩罚控制动作输出的突变，让电机指令更平滑，防止抖动
        self.rewards.dof_acc_l2.weight = -1.25e-7      # 惩罚过大的关节加速度
