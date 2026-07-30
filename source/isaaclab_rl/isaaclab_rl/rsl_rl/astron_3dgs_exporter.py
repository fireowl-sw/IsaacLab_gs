import os
import json
import mmap
import struct
import socket
import numpy as np
import torch
from pathlib import Path

MAGIC = 0x3D650001

class Astron3DGSExporter:
    """
    Real-time Pose Exporter for Walker Astron.
    Extracts link and camera transforms on GPU, writes to file-mapped shared memory,
    and sends a UDP trigger pulse to host renderer.
    """
    def __init__(self, env, max_envs: int = 64, max_links: int = 64):
        self.env = env
        self.num_envs = min(env.num_envs, max_envs)
        self.max_envs = max_envs
        self.max_links = max_links
        self.device = env.device
        
        # 1. 建立共享文件 (POSIX MMAP) 在共享挂载目录下
        self.shm_dir = Path("/workspace/IsaacLab_gs/logs")
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        self.shm_file = self.shm_dir / "robot_3dgs_shm"
        
        # 2. 计算并预分配共享内存文件的大小 (约 270 KB)
        # Header (64 bytes) + Metadata (4096 bytes) + Links (max_envs * max_links * 16 * 4) + Cams (max_envs * 16 * 4) + Intrinsics (max_envs * 9 * 4)
        self.links_offset = 64 + 4096
        self.cams_offset = self.links_offset + (self.max_envs * self.max_links * 16 * 4)
        self.intrinsics_offset = self.cams_offset + (self.max_envs * 16 * 4)
        self.total_size = self.intrinsics_offset + (self.max_envs * 9 * 4)
        
        # 创建空洞文件以锁定大小
        with open(self.shm_file, "wb") as f:
            f.truncate(self.total_size)
            
        self.file_obj = open(self.shm_file, "r+b")
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), self.total_size)
        
        # 3. 初始化 UDP 客户端发送网络脉冲 (使用 host 网络 127.0.0.1 通信)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_addr = ("127.0.0.1", 12347)
        
        # 4. 获取机器人 link 名字映射
        self.robot = env.scene["robot"]
        self.link_names = self.robot.data.body_names
        
        # 写入 Metadata JSON
        meta_dict = {
            "link_names": self.link_names,
            "camera_name": "head_stereo_left_Camera"
        }
        meta_bytes = json.dumps(meta_dict).encode("utf-8")
        if len(meta_bytes) > 4096:
            meta_bytes = meta_bytes[:4096]
        else:
            meta_bytes = meta_bytes + b"\x00" * (4096 - len(meta_bytes))
            
        self.mmap_obj.seek(64)
        self.mmap_obj.write(meta_bytes)
        
        # 5. 预设相机内参 (用于高斯投影光栅化)
        self.width = 1920
        self.height = 1080
        fx = (50.0 / 36.0) * self.width
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        # 每个环境共享这套内参
        intrinsics = np.zeros((self.max_envs, 3, 3), dtype=np.float32)
        for i in range(self.max_envs):
            intrinsics[i] = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ]
        self.mmap_obj.seek(self.intrinsics_offset)
        self.mmap_obj.write(intrinsics.tobytes())
        
        self.frame_idx = 0
        print(f"[3DGS Exporter] Initialized with {self.num_envs} envs, writing to {self.shm_file}")

    def quat_to_matrix_numpy(self, quats, pos):
        """批量将四元数与位置转化为 4x4 变换矩阵 (NumPy 加速)"""
        N = quats.shape[0]
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        R = np.zeros((N, 4, 4), dtype=np.float32)
        R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        R[:, 0, 3] = pos[:, 0]
        
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
        R[:, 1, 2] = 2*y*z - 2*w*x
        R[:, 1, 3] = pos[:, 1]
        
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 2] = 1 - 2*x**2 - 2*y**2
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 3] = pos[:, 2]
        
        R[:, 3, 3] = 1.0
        return R

    def step(self):
        """在仿真 Step 后，抓取显存位姿，写入 MMAP 并发送 UDP 信号"""
        body_pos_w = self.robot.data.body_pos_w[:self.num_envs].cpu().numpy()
        body_quat_w = self.robot.data.body_quat_w[:self.num_envs].cpu().numpy()
        
        links_mat = np.zeros((self.max_envs, self.max_links, 4, 4), dtype=np.float32)
        for env_idx in range(self.num_envs):
            mats = self.quat_to_matrix_numpy(body_quat_w[env_idx], body_pos_w[env_idx])
            links_mat[env_idx, :len(self.link_names)] = mats
            
        self.mmap_obj.seek(self.links_offset)
        self.mmap_obj.write(links_mat.tobytes())
        
        cams_mat = np.zeros((self.max_envs, 4, 4), dtype=np.float32)
        root_pos = self.robot.data.root_pos_w[:self.num_envs].cpu().numpy()
        
        for env_idx in range(self.num_envs):
            base_pos = root_pos[env_idx]
            cam_pos = base_pos + np.array([-2.2, -1.8, 0.75])
            target = base_pos + np.array([0.0, 0.0, 0.2])
            
            forward = target - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, [0.0, 0.0, 1.0])
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            cam_matrix = np.eye(4, dtype=np.float32)
            cam_matrix[:3, 0] = right
            cam_matrix[:3, 1] = up
            cam_matrix[:3, 2] = -forward
            cam_matrix[:3, 3] = cam_pos
            
            cams_mat[env_idx] = cam_matrix
            
        self.mmap_obj.seek(self.cams_offset)
        self.mmap_obj.write(cams_mat.tobytes())
        
        sim_time = self.env.sim.current_time if hasattr(self.env.sim, "current_time") else 0.0
        header_bytes = struct.pack("!IIId", MAGIC, self.frame_idx, self.num_envs, sim_time)
        self.mmap_obj.seek(0)
        self.mmap_obj.write(header_bytes)
        
        try:
            self.sock.sendto(header_bytes, self.target_addr)
        except Exception:
            pass
            
        self.frame_idx += 1

    def close(self):
        """释放资源"""
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.file_obj:
            self.file_obj.close()
        print("[3DGS Exporter] File mapping resources released.")
