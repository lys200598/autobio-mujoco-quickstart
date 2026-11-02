# AutoBio Pickup：环境、数据集与一次 MuJoCo 渲染

本目录记录已经验证过的最小流程：安装 AutoBio Pickup 所需环境，下载公开的
`pickup-mujoco` LeRobot 数据集，并把其中一个 episode 在本地 MuJoCo 场景中渲染为 MP4。

> 第 1 节从包含本目录的父目录开始执行，并创建/进入 `autobio-mujoco-quickstart`；后续各节
> 默认已经位于该目录，因此均以 `cd .` 开头。本文只提供命令；请由使用者自行执行。

## 1. 环境配置

AutoBio 当前使用 Linux、Python 3.11、MuJoCo 3.3.0 和 `mjlab` 物理插件。环境配置的
关键顺序是：**先获得 AutoBio 源码和插件，再创建 Conda 环境，最后安装 Python 包**。
不能只创建 `.venv`，因为第 3 步还需要源码中的模型、网格和插件。

```bash
cd autobio-mujoco-quickstart

mkdir -p .vendor
git clone https://github.com/autobio-bench/AutoBio.git .vendor/AutoBio

test -f .vendor/AutoBio/autobio/libmjlab.so.3.3.0

conda create -y -p .vendor/AutoBio/.venv python=3.11 pip

.vendor/AutoBio/.venv/bin/python -m pip install -i https://pypi.doubanio.com/simple \
  "mujoco==3.3.0" numpy scipy "jax[cpu]" toppra trimesh shapely triangle \
  manifold3d sympy zstandard tqdm networkx usd-core ffmpeg "imageio[ffmpeg]" \
  matplotlib scikit-image pyarrow av huggingface_hub
```

验证基础依赖：

```bash
.vendor/AutoBio/.venv/bin/python -c "import numpy, mujoco; print(numpy.__version__, mujoco.__version__)"
```

预期 MuJoCo 版本为 `3.3.0`。请勿把 `.vendor/AutoBio/.venv/bin/python` 先 `resolve()` 成
基础解释器路径；这样会绕过 Conda 环境的 `site-packages`。

## 2. 下载公开数据集

官方公开的 Pickup MuJoCo 数据集是 `autobio-bench/pickup-mujoco`，格式为 LeRobot v2：
100 条 episode、50 fps、两路 224×224 AV1 视频，以及每帧 7 维机器人 state/action。

```bash
cd .

.vendor/AutoBio/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='autobio-bench/pickup-mujoco', repo_type='dataset', local_dir='data/pickup-mujoco')"
```

下载完成后，目录应包含：

```text
data/pickup-mujoco/
  data/chunk-000/episode_000000.parquet
  meta/info.json
  videos/chunk-000/image/episode_000000.mp4
  videos/chunk-000/wrist_image/episode_000000.mp4
```

## 3. 渲染一个已下载 episode

本目录已内置最小的 `grasp_execution_pipeline` Python 包，因此其他用户不需要取得本项目
根目录中的同名包。以下命令进入 quickstart 目录后，使用 AutoBio 官方 `pickup.xml`、
`libmjlab` 插件和公开数据集中的机器人 state 渲染 episode 0：

```bash
cd .

.vendor/AutoBio/.venv/bin/python -m grasp_execution_pipeline.datasets.render_autobio_pickup \
  --dataset-root data/pickup-mujoco \
  --episode 0 \
  --output outputs/autobio_pickup_ep000000.mp4 \
  --model .vendor/AutoBio/autobio/model/scene/pickup.xml \
  --plugin-library .vendor/AutoBio/autobio/libmjlab.so.3.3.0 \
  --camera table_cam_front \
  --autobio-pickup-seed 0 \
  --gl egl
```

输出文件是：

```text
outputs/autobio_pickup_ep000000.mp4
```

### 重要限制

公开 LeRobot 数据只包含 6 个 ALOHA 关节和 1 个夹爪开度，**没有**试管的逐帧
`qpos/qvel`、接触和完整 `ctrl`。上面的命令会按 episode seed 将试管初始化到正确机架
槽位，并回放机械臂姿态；它不能精确恢复原视频中试管被夹起后的物理运动。

## 4. 生成并渲染完整物理轨迹（推荐用于可复现抓取）

若目标是获得会随夹爪真实运动的试管轨迹，请让 AutoBio 官方专家重新生成仿真数据：

```bash
cd .vendor/AutoBio/autobio

../.venv/bin/python pickup_centrifuge_tube.py
bash render.bash logs/pickup
```

这会创建 `logs/pickup/<timestamp>/`。每条轨迹包含：

```text
states.npy.zst                  # 完整 MuJoCo 状态：qpos、qvel、ctrl、外力等
info.json                       # 状态切片与相机映射
table_cam_front.mp4
table_cam_left.mp4
1-aloha-wrist_cam_left.mp4
```

批次共享的 `logs/pickup/model.mjb` 与每条轨迹的完整状态可用于逐帧精确渲染和后续
数据格式转换。

## 故障排查

- `No module named pip`：重新以 `conda create -p .vendor/AutoBio/.venv python=3.11 pip` 创建环境。
- `No module named numpy`：确认命令实际调用的是
  `.vendor/AutoBio/.venv/bin/python`，不要使用其符号链接解析后的基础解释器。
- `插件不存在: .vendor/AutoBio/autobio/libmjlab.so.3.3.0`：第 1 节未完成。请检查
  `.vendor/AutoBio` 是否为完整 Git 源码目录；若它只有 `.venv`，执行本节的
  “修复已存在但只有 `.venv` 的目录”。README 不会自动删除已有目录。
- `EGL`/OpenGL 错误：无显示器服务器优先加 `--gl egl`，并确认系统 EGL 驱动可用。
- 渲染中试管没有被夹起：这是公开 LeRobot 数据缺少物体状态的预期限制；改用第 4 节生成的
  完整日志，或取得原始 `states.npy.zst`。

## 参考

- [AutoBio 官方仓库](https://github.com/autobio-bench/AutoBio)
- [官方 Pickup MuJoCo 数据集](https://huggingface.co/datasets/autobio-bench/pickup-mujoco)
