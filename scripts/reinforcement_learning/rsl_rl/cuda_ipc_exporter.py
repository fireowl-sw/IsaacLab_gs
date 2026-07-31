import os
import sys
import json
import mmap
import struct
import socket
import numpy as np
import torch
from pathlib import Path

from isaaclab.utils.math import matrix_from_quat

class CudaIpcExporter:
    """
    Isaac Lab 侧高效 GPU-Vectorized IPC / Shared Memory (SHM) 数据导出器。
    
    采用共享内存 (Memory-Mapped File) 实现 0 拷贝数据传递：
    1. 预先分配共享内存块 (mmap)。
    2. 每帧将 N 个并发环境的 4x4 连杆矩阵与相机矩阵在 GPU 上批量计算并写入 mmap。
    3. 通过极轻量 UDP 端口 12347 发送 32 字节心跳信号通知宿主机渲染端。
    """
    MAGIC = 0x3D650001  # 校验头魔数

    def __init__(self, stage, config_path: str = "config.json"):
        self.stage = stage
        self.config_path = Path(config_path)
        
        # 默认配置参数
        self.max_envs = 64
        self.max_links = 64
        self.shm_name = "robot_3dgs_shm"
        self.signal_port = 12347
        self.camera_name = "camera"
        self.host = "127.0.0.1"
        self.shm_file_path = self.config_path.resolve().parent / self.shm_name
        self.active_env_idx = 0
        
        self._load_config()
        self.shm_size = self.calculate_shm_size()
        self.mmap_obj = None
        self.file_obj = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.frame_idx = 0
        self.link_names = []
        self.link_name_to_idx = {}
        self.camera_prim = None
        
    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    ipc_cfg = cfg.get("ipc", {})
                    self.max_envs = ipc_cfg.get("max_envs", self.max_envs)
                    self.max_links = ipc_cfg.get("max_links", self.max_links)
                    self.shm_name = ipc_cfg.get("shm_name", self.shm_name)
                    self.signal_port = ipc_cfg.get("signal_port", self.signal_port)
                    
                    rendering_cfg = cfg.get("rendering", {})
                    self.camera_name = rendering_cfg.get("camera_name", self.camera_name)
                    self.active_env_idx = rendering_cfg.get("active_env_idx", self.active_env_idx)
                    self.shm_file_path = self.config_path.resolve().parent / self.shm_name
                    print(f"[CudaIpcExporter] Loaded config from '{self.config_path}' -> active_env_idx: {self.active_env_idx}, camera_name: '{self.camera_name}'")
            except Exception as e:
                print(f"[CudaIpcExporter] Config load warning: {e}")

    def calculate_shm_size(self):
        # 头部: magic(4), frame_idx(4), num_envs(4), num_links(4), timestamp(8), reserved(40) = 64 字节
        header_size = 64
        # 连杆映射元数据 (JSON UTF-8 char buffer): 4096 字节
        metadata_size = 4096
        # links 矩阵: max_envs * max_links * 16 * 4 字节
        links_size = self.max_envs * self.max_links * 16 * 4
        # cams 矩阵: max_envs * 16 * 4 字节
        cams_size = self.max_envs * 16 * 4
        # intrinsics 矩阵: max_envs * 9 * 4 字节
        intrinsics_size = self.max_envs * 9 * 4
        
        return header_size + metadata_size + links_size + cams_size + intrinsics_size

    def start(self):
        # 初始化/创建共享内存文件
        self.shm_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_obj = open(self.shm_file_path, "wb+")
        self.file_obj.truncate(self.shm_size)
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), self.shm_size)
        
        # 强制将 Fabric 物理结果写回 USD Stage (为了读取相机外参)
        try:
            import carb
            settings = carb.settings.get_settings()
            for key in ["/physics/updateToUsd", "/persistent/physics/updateToUsd", "/physx/updateToUsd"]:
                settings.set(key, True)
        except Exception as e:
            print(f"[CudaIpcExporter] Warning: Failed to set updateToUsd: {e}")

        print(f"[CudaIpcExporter] Shared Memory (SHM) Exporter started at: {self.shm_file_path} ({self.shm_size / (1024*1024):.2f} MB)")

    def _discover_camera(self):
        """寻找配置的相机 Prim"""
        if self.stage is None:
            return None
        
        target_cam = self.camera_name.lower()
        from pxr import UsdGeom
        
        # 1. 尝试模糊匹配名称
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Camera":
                pname = prim.GetName().lower()
                ppath = prim.GetPath().pathString.lower()
                clean_target = target_cam.replace("_camera", "").strip()
                if (target_cam in pname) or (target_cam in ppath) or (clean_target in pname) or (clean_target in ppath):
                    print(f"[CudaIpcExporter] Camera matched: '{prim.GetPath().pathString}'")
                    return prim
        
        # 2. 备选降级：抓取第一个有效相机
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Camera":
                ppath = prim.GetPath().pathString.lower()
                if not any(sys_c in ppath for sys_c in ["/omniversekit_", "/front", "/top", "/right"]):
                    print(f"[CudaIpcExporter] Camera fallback to: '{prim.GetPath().pathString}'")
                    return prim
        
        return None

    def export_tensors(self, env):
        # 定期热加载 config.json 以允许动态更新 active_env_idx 或 camera_name
        if self.frame_idx % 100 == 0:
            self._load_config()

        num_envs = min(env.unwrapped.num_envs, self.max_envs)
        sim_time = env.unwrapped.sim.get_physics_dt() * self.frame_idx # 模拟时间

        # 首次发现关节列表元数据并写入 SHM
        if len(self.link_names) == 0:
            self.link_names = env.unwrapped.scene["robot"].data.body_names
            # 做一次静态对齐校验前验
            print("\n" + "="*50)
            print("[CudaIpcExporter] Pre-verification link alignment:")
            for idx, name in enumerate(self.link_names):
                print(f"  - Index {idx:02d}: '{name}'")
            print("="*50 + "\n")
            
            metadata_json = json.dumps({
                "link_names": self.link_names,
                "camera_name": self.camera_name
            }).encode("utf-8")
            self.mmap_obj.seek(64)
            self.mmap_obj.write(metadata_json.ljust(4096, b"\x00"))

        num_links = min(len(self.link_names), self.max_links)

        # 1. 批量在 GPU 上合成仿射变换矩阵
        # pos shape: [num_envs, num_links, 3]
        # quat shape: [num_envs, num_links, 4] (wxyz format)
        pos = env.unwrapped.scene["robot"].data.body_pos_w[:, :num_links]
        quat = env.unwrapped.scene["robot"].data.body_quat_w[:, :num_links]
        
        # 转换为 xyzw 并生成旋转矩阵 [num_envs, num_links, 3, 3]
        quat_xyzw = quat.roll(-1, dims=-1)
        rotation_gpu = matrix_from_quat(quat_xyzw)

        # 初始化 homogeneous 变换矩阵并向量化填充
        mats_gpu = torch.zeros((num_envs, num_links, 4, 4), dtype=torch.float32, device=pos.device)
        # 按照 USD 行优先存储规则：mats_gpu[..., :3, :3] = R.T (即 columns 旋转)，mats_gpu[..., 3, :3] = T
        mats_gpu[..., :3, :3] = rotation_gpu.transpose(-1, -2)
        mats_gpu[..., 3, :3] = pos
        mats_gpu[..., 3, 3] = 1.0

        # 将 GPU 张量拉回到 CPU numpy
        links_mat_np = mats_gpu.cpu().numpy()

        # 2. 构建相机外参及内参矩阵
        cams_mat = np.zeros((self.max_envs, 4, 4), dtype=np.float32)
        intrinsics_mat = np.zeros((self.max_envs, 3, 3), dtype=np.float32)
        for env_i in range(self.max_envs):
            cams_mat[env_i] = np.eye(4, dtype=np.float32)
            intrinsics_mat[env_i] = np.eye(3, dtype=np.float32)

        # 延迟发现并更新相机
        if self.camera_prim is None:
            self.camera_prim = self._discover_camera()

        if self.camera_prim is not None:
            from pxr import UsdGeom
            camera_geom = UsdGeom.Camera(self.camera_prim)
            world_transform = camera_geom.ComputeLocalToWorldTransform(sim_time)
            cam_mat = np.array(world_transform, dtype=np.float32)
            # 广播/填充相机外参
            for env_i in range(num_envs):
                cams_mat[env_i] = cam_mat
                
            fl = camera_geom.GetFocalLengthAttr().Get(sim_time) or 50.0
            ha = camera_geom.GetHorizontalApertureAttr().Get(sim_time) or 36.0
            for env_i in range(num_envs):
                intrinsics_mat[env_i, 0, 0] = float(fl)
                intrinsics_mat[env_i, 1, 1] = float(ha)

        # 写入共享内存的 offset 映射
        metadata_offset = 64
        links_offset = metadata_offset + 4096
        cams_offset = links_offset + (self.max_envs * self.max_links * 16 * 4)
        intrinsics_offset = cams_offset + (self.max_envs * 16 * 4)

        # 组装完整的 links 矩阵 buffer（对于未使用的 env 部分保留默认值 0）
        full_links_mat = np.zeros((self.max_envs, self.max_links, 4, 4), dtype=np.float32)
        # 预填单位矩阵
        for env_i in range(self.max_envs):
            for link_i in range(self.max_links):
                full_links_mat[env_i, link_i] = np.eye(4, dtype=np.float32)
        
        full_links_mat[:num_envs, :num_links] = links_mat_np

        # 写入矩阵数据到 mmap
        self.mmap_obj.seek(links_offset)
        self.mmap_obj.write(full_links_mat.tobytes())
        self.mmap_obj.seek(cams_offset)
        self.mmap_obj.write(cams_mat.tobytes())
        self.mmap_obj.seek(intrinsics_offset)
        self.mmap_obj.write(intrinsics_mat.tobytes())

        # 写入头部标志
        self.frame_idx += 1
        header_data = struct.pack(
            "=IIIId40x",
            self.MAGIC,
            self.frame_idx,
            num_envs,
            num_links,
            sim_time
        )
        self.mmap_obj.seek(0)
        self.mmap_obj.write(header_data)

        # 4. 发送 32 字节心跳 UDP 握手包到宿主机播放器
        signal_pkt = struct.pack("!IIId", self.MAGIC, self.frame_idx, num_envs, sim_time)
        try:
            self.sock.sendto(signal_pkt, (self.host, self.signal_port))
        except Exception:
            pass

    def stop(self):
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file_obj is not None:
            self.file_obj.close()
        print("[CudaIpcExporter] Exporter stopped and resources cleaned up.")


def setup_ipc_exporter(stage, config_path: str = "config.json"):
    exporter = CudaIpcExporter(stage, config_path)
    exporter.start()
    return exporter
