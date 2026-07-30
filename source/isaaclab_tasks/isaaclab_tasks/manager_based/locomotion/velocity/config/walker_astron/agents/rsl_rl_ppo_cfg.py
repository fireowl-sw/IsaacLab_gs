# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO (Proximal Policy Optimization - 近端策略优化) 强化学习网络训练超参数配置文件。
本文件使用了 RSL-RL 库提供的强化学习算法框架，专为 Walker Astron 机器人在平地上快速收敛行走而设计。
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WalkerAstronFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    RSL-RL 算法运行器（Runner）的核心配置参数。
    此配置决定了训练循环的步长、迭代次数、神经网络的隐层结构和学习率等关键参数。
    """
    num_steps_per_env = 24       # 每个并发环境每次迭代时采集的样本步数（步长越多，梯度估算越准，但训练步进越慢）
    max_iterations = 1500        # 最大训练迭代代数（PPO 主循环的上限，1500 代在平地下通常需要 15 到 30 分钟）
    save_interval = 100          # 每训练 100 代自动在本地保存一次神经网络模型权重文件 (.pt)
    experiment_name = "walker_astron_flat" # 实验项目名称，会作为文件夹名称存放在本地 logs 目录中
    
    # ---------------------------------------------
    # 1. 演员网络配置 (Actor - 负责根据状态输出关节目标位置指令的神经网络)
    # ---------------------------------------------
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128], # 三层全连接多层感知机 (MLP) 隐藏层，每层包含 128 个神经元。
                                     # 128神经元对于单纯的平地关节运动控制而言，体量小、推理速度极快，防过拟合效果好。
        activation="elu",            # 隐层激活函数，采用指数线性单元 (ELU)，在机器人控制中梯度传导极佳。
        obs_normalization=True,      # 启用观测值（Observation）的在线均值/标准差归一化，能大幅提高收敛速度与平滑度。
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0), # 高斯分布探索噪声的初始标准差，值越大开局探索越剧烈。
    )
    
    # ---------------------------------------------
    # 2. 评论家网络配置 (Critic - 评估当前机器人身体姿态是安全还是危险的评估器网络)
    # ---------------------------------------------
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128], # 结构与 Actor 保持对称
        activation="elu",
        obs_normalization=True,
    )
    
    # ---------------------------------------------
    # 3. PPO 优化算法的核心参数配置 (PPO Core Hyperparameters)
    # ---------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # 状态价值损失系数
        use_clipped_value_loss=True,   # 启用裁剪的价值损失，防止 Critic 大脑波动剧烈
        clip_param=0.2,                # 策略更新幅度裁剪限制（防止新老策略偏离过大导致步态训练直接崩溃）
        entropy_coef=0.01,             # 熵系数（鼓励策略随机探索，防止过早陷入局部最优而不敢跨步）
        num_learning_epochs=5,         # 每次采集完数据后，利用当前数据重复训练更新网络的轮数
        num_mini_batches=4,            # 将单次采样的总数据切分为 4 个 Mini-batch 进行小批次梯度下降更新
        learning_rate=1.0e-3,          # 优化器的学习率 (Learning Rate / LR)
        schedule="adaptive",           # 启用自适应学习率机制：算法会根据 KL 散度的实际变化，自动调小或调大 LR。
        gamma=0.99,                    # 长期奖励衰减因子 (Discount Factor)：0.99 代表非常看重长远回报（如平稳往前走）。
        lam=0.95,                      # GAE (广义优势估算) 偏差与方差折中因子
        desired_kl=0.01,               # 设定的理想 KL 散度约束阈值（自适应 LR 调整的重要依据）
        max_grad_norm=1.0,             # 梯度裁剪最大模长限制，防梯度爆炸。
    )
