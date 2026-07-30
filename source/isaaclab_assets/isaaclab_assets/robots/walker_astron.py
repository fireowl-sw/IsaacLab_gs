# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walker Astron 双足人形机器人的 Isaac Lab 资产配置文件。"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# 机器人用户配置 (Articulation Definition)
##

WALKER_ASTRON_CFG = ArticulationCfg(
    # 1. 场景生成配置 (Spawn properties)
    spawn=sim_utils.UsdFileCfg(
        # 机器人 USD 模型路径 (已挂载在容器内的持久化路径)
        usd_path="/home/ubt/isaacsim_ws/isaacdata/0.ISAAC/astron_usd/Collected_walker_astron_v1_mat/walker_astron_v1flat.usd",
        # 开启接触传感器 (Contact Sensors)，用于足底碰撞检测和步态训练
        activate_contact_sensors=True,
        # 刚体物理属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,          # 不禁用重力
            retain_accelerations=False,     # 物理步进时不强制保留上一步加速度
            linear_damping=0.0,             # 线性阻尼
            angular_damping=0.0,            # 旋转阻尼
            max_linear_velocity=1000.0,     # 最大线速度限制 (防止爆数值)
            max_angular_velocity=1000.0,    # 最大角速度限制
            max_depenetration_velocity=1.0, # 最大重叠穿透回弹速度 (防止刚体碰撞时飞天)
        ),
        # 关节根部解算器配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,             # 禁用自身碰撞 (防止双腿内侧自己撞自己导致卡死)
            solver_position_iteration_count=4,       # 位置解算器迭代次数 (默认 4)
            solver_velocity_iteration_count=4,       # 速度解算器迭代次数 (默认 4)
        ),
    ),
    
    # 2. 初始状态配置 (Initial State - 每次 Episode 开始时机器人的姿态)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0), # 躯干质心初始高度 (单位: 米)
        # 默认关节角度 (单位: 弧度)。采用微蹲 (Micro Squat) 姿态可以大幅提高开局的物理稳定性
        joint_pos={
            # === 左腿关节 (Left Leg) ===
            "L_hip_roll_joint": 0.0,      # 髋关节翻滚角 (侧摆)
            "L_hip_yaw_joint": 0.0,       # 髋关节航向角 (偏航)
            "L_hip_pitch_joint": -0.1,    # 髋关节俯仰角 (前屈)
            "L_knee_pitch_joint": 0.2,     # 膝关节俯仰角 (屈膝)
            "L_ankle_pitch_joint": -0.1,   # 踝关节俯仰角 (勾脚)
            "L_ankle_roll_joint": 0.0,     # 踝关节翻滚角
            
            # === 右腿关节 (Right Leg) ===
            "R_hip_roll_joint": 0.0,
            "R_hip_yaw_joint": 0.0,
            "R_hip_pitch_joint": -0.1,
            "R_knee_pitch_joint": 0.2,
            "R_ankle_pitch_joint": -0.1,
            "R_ankle_roll_joint": 0.0,
            
            # === 腰部关节 (Waist) ===
            "waist_pitch_joint": 0.06,     # 腰部俯仰（微屈前倾 3.4 度以维持质心前移，防止后仰）
            "waist_yaw_joint": 0.0,       # 腰部偏航
            "waist_roll_joint": 0.0,      # 腰部侧摆

            # === 左臂关节 (Left Arm) ===
            "L_shoulder_pitch_joint": 0.0, # 肩部俯仰
            "L_shoulder_roll_joint": 0.0,   # 肩部外展
            "L_shoulder_yaw_joint": 0.0,    # 肩部旋转
            "L_elbow_pitch_joint": 0.0,    # 肘部屈伸
            "L_elbow_yaw_joint": 0.0,       # 肘部旋转
            "L_wrist_roll_joint": 0.0,      # 腕部翻滚
            "L_wrist_pitch_joint": 0.0,     # 腕部俯仰

            # === 右臂关节 (Right Arm) ===
            "R_shoulder_pitch_joint": 0.0,
            "R_shoulder_roll_joint": 0.0,
            "R_shoulder_yaw_joint": 0.0,
            "R_elbow_pitch_joint": 0.0,
            "R_elbow_yaw_joint": 0.0,
            "R_wrist_roll_joint": 0.0,
            "R_wrist_pitch_joint": 0.0,

            # === 头部关节 (Head) ===
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0}, # 所有关节初始速度归零
    ),
    
    # 软性关节限位因子 (Soft Limit Factor): 允许关节运动到硬性几何极限的 90% 处，防止发生物理碰撞爆破
    soft_joint_pos_limit_factor=0.9,
    
    # 3. 驱动器配置 (使用从公司 Astron 神经网络部署控制器中提取的真实物理调优 PD 增益)
    actuators={
        # === 髋部侧摆与俯仰、膝关节、腰部电机组 (主要承重与大范围运动) ===
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "[LR]_hip_roll_joint",
                "[LR]_hip_pitch_joint",
                "[LR]_knee_pitch_joint",
                "waist_pitch_joint",
                "waist_roll_joint"
            ],
            effort_limit_sim=300, # 电机最大输出力矩限制 (N·m)
            stiffness={
                "[LR]_hip_pitch_joint": 150.0,   # 髋部俯仰控制重心前后
                "[LR]_hip_roll_joint": 110.0,    # 髋部侧摆控制侧向跨步
                "[LR]_knee_pitch_joint": 200.0,   # 膝部大负荷支撑关节
                "waist_pitch_joint": 180.0,       # 腰部俯仰控制上半身直立
                "waist_roll_joint": 150.0,        # 腰部翻滚稳定左/右平衡
            },
            damping={
                "[LR]_hip_pitch_joint": 10.0,
                "[LR]_hip_roll_joint": 10.0,
                "[LR]_knee_pitch_joint": 10.0,
                "waist_pitch_joint": 5.0,
                "waist_roll_joint": 3.0,
            },
        ),
        
        # === 髋部偏航、腰部偏航关节 (低负荷旋转定位) ===
        "hip_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                "[LR]_hip_yaw_joint",
                "waist_yaw_joint"
            ],
            effort_limit_sim=100,
            stiffness={
                "[LR]_hip_yaw_joint": 50.0,     # 控制腿部内八/外八
                "waist_yaw_joint": 80.0,        # 控制上半身左右扭腰
            },
            damping={
                "[LR]_hip_yaw_joint": 5.0,
                "waist_yaw_joint": 4.0,
            },
        ),
        
        # === 脚踝电机组 (Ankles - 高精度柔性稳定支撑) ===
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_ankle_.*_joint"],
            effort_limit_sim=100, # 踝部电机最大扭矩
            stiffness={
                "[LR]_ankle_pitch_joint": 45.0,  # 踝关节勾脚/踩地刚度
                "[LR]_ankle_roll_joint": 40.0,   # 踝关节内外摆刚度
            },
            damping={
                "[LR]_ankle_pitch_joint": 1.5,
                "[LR]_ankle_roll_joint": 1.5,
            },
        ),
        
        # === 上肢手臂与头部电机组 (Arms & Head) ===
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "[LR]_shoulder_.*_joint",
                "[LR]_elbow_.*_joint",
                "[LR]_wrist_.*_joint",
                "head_.*_joint"
            ],
            effort_limit_sim=100,
            stiffness={
                "[LR]_shoulder_.*_joint": 80.0,  # 肩部电机防手臂甩荡
                "[LR]_elbow_.*_joint": 50.0,     # 肘部电机
                "[LR]_wrist_.*_joint": 40.0,     # 腕部电机
                "head_.*_joint": 40.0,           # 头部关节
            },
            damping={
                "[LR]_shoulder_.*_joint": 2.5,
                "[LR]_elbow_.*_joint": 1.5,
                "[LR]_wrist_.*_joint": 1.5,
                "head_.*_joint": 1.0,
            },
        ),
    },
)
"""Walker Astron 双足人形机器人的完整真实物理调优资产配置常数。"""
