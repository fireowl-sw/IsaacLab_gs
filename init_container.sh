#!/bin/bash
# Isaac Lab 容器一键初始化配置脚本

echo "=== 开始配置容器内部环境 ==="

# 1. 设置环境变量
export ISAACSIM_PATH=/isaac-sim
echo "export ISAACSIM_PATH=/isaac-sim" >> ~/.bashrc
echo "export PATH=/isaac-sim:\$PATH" >> ~/.bashrc

# 2. 创建 python3 包装脚本
echo "=== 正在创建 python3 包装器 ==="
cat << 'EOF' > /usr/local/bin/python3
#!/bin/bash
exec /isaac-sim/python.sh "$@"
EOF
chmod +x /usr/local/bin/python3

# 3. 配置 pip 清华源
echo "=== 正在配置 pip 清华源 ==="
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 4. 配置 Ubuntu apt 软件源为清华镜像源 (加速 apt-get install)
echo "=== 正在配置 Ubuntu apt 清华源 ==="
if [ -f /etc/apt/sources.list ]; then
    sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
fi
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources
fi

# 5. 更新 pip
python3 -m pip install --upgrade pip

# 6. 安装容器内缺失的 git 依赖
if ! command -v git &> /dev/null; then
    echo "=== 正在安装系统 git 依赖 ==="
    apt-get update && apt-get install -y git
fi

# 7. 执行 IsaacLab 绑定安装
echo "=== 正在执行 Isaac Lab 绑定安装 ==="
./isaaclab.sh --install

echo "=== 配置完成！你可以运行仿真了 ==="
