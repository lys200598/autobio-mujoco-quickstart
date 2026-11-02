"""将公开 AutoBio Pickup LeRobot episode 渲染到 AutoBio MuJoCo 场景。

此文件刻意不依赖原项目的 ``grasp_execution_pipeline``：它是 quickstart 随附的最小
可分发入口，读取 7 维 LeRobot state/action，加载官方 pickup.xml，并输出 H.264 MP4。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


ARM_JOINT_CANDIDATES = (
    ("left/waist", "1/aloha:left/waist"),
    ("left/shoulder", "1/aloha:left/shoulder"),
    ("left/elbow", "1/aloha:left/elbow"),
    ("left/forearm_roll", "1/aloha:left/forearm_roll"),
    ("left/wrist_angle", "1/aloha:left/wrist_angle"),
    ("left/wrist_rotate", "1/aloha:left/wrist_rotate"),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="渲染一个公开 AutoBio Pickup LeRobot episode")
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--episode", type=int, default=0)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--plugin-library", type=Path, required=True)
    result.add_argument("--camera", default="table_cam_front")
    result.add_argument("--autobio-pickup-seed", type=int, default=None)
    result.add_argument("--source", choices=("state", "action"), default="state")
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default=None)
    return result


def _read_episode(root: Path, episode: int, source: str) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 pyarrow；请用 venv Python 安装 pyarrow") from error

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"找不到数据集元数据: {info_path}。"
            "请在 autobio-mujoco-quickstart 目录执行，并传入 "
            "--dataset-root data/pickup-mujoco；不要使用旧路径 data/autobio/pickup-mujoco。"
        )
    info = json.loads(info_path.read_text())
    chunk_size = int(info["chunks_size"])
    chunk = episode // chunk_size
    parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"找不到 episode: {parquet_path}")
    column = "state" if source == "state" else "actions"
    table = parquet.read_table(parquet_path, columns=[column, "timestamp"])
    states = np.asarray(table.column(column).to_pylist(), dtype=np.float64)
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 7:
        raise RuntimeError(f"{column} 应为 (T, 7)，实际为 {states.shape}")
    if timestamps.shape != (states.shape[0],):
        raise RuntimeError("timestamp 与状态帧数不一致")
    return states[:, :6], states[:, 6], float(info["fps"])


def _load_mujoco(gl: str | None, plugin: Path):
    if gl:
        os.environ["MUJOCO_GL"] = gl
    elif not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 mujoco；请安装 mujoco==3.3.0") from error
    if not plugin.is_file():
        raise FileNotFoundError(
            f"插件不存在: {plugin}。请确认已将 AutoBio 源码克隆到 "
            ".vendor/AutoBio（不能只创建 .vendor/AutoBio/.venv）。"
        )
    mujoco.mj_loadPluginLibrary(str(plugin.resolve()))
    return mujoco


def _joint_id(mujoco, model, candidates: tuple[str, ...]) -> int:
    for name in candidates:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            return joint_id
    raise RuntimeError(f"找不到关节；候选名称: {candidates}")


def _initialize_pickup(mujoco, model, data, seed: int) -> tuple[int, int]:
    """按官方 Pickup.reset 的随机数顺序复原试管初始 rack 槽位。"""
    rng = np.random.RandomState(seed)
    rng.uniform((-0.2, 0, 0, -0.05, 0, -0.1), (0.2, 0.3, 0.1, 0.05, 0.3, 0.1))
    row, column = int(rng.randint(2)), int(rng.randint(5))
    mujoco.mj_forward(model, data)
    site_id = -1
    for index in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index) or ""
        if name.startswith("/") and len(name.split("-")) > 1 and name.split("-")[1] == "50ml":
            site_id = index
            break
    if site_id < 0:
        raise RuntimeError("场景中没有 AutoBio 50ml rack site")
    user = model.site_user[site_id]
    position = data.site_xpos[site_id].copy()
    axes = data.site_xmat[site_id].reshape(3, 3)
    position += row * user[2] * axes[:, 0] + column * user[4] * axes[:, 1]
    for name in ("centrifuge_50ml_screw_cap", "centrifuge_50ml_screw_body"):
        joint_id = _joint_id(mujoco, model, (name,))
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address : address + 3] = position
    mujoco.mj_forward(model, data)
    return row, column


def run(args: argparse.Namespace) -> None:
    arm_states, gripper_states, fps = _read_episode(
        args.dataset_root.expanduser().resolve(), args.episode, args.source
    )
    mujoco = _load_mujoco(args.gl, args.plugin_library.expanduser())
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if args.autobio_pickup_seed is None:
        mujoco.mj_resetData(model, data)
    else:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        row, column = _initialize_pickup(mujoco, model, data, args.autobio_pickup_seed)
        print(f"AutoBio Pickup reset: seed={args.autobio_pickup_seed}, rack=({row}, {column})")

    arm_qpos = [int(model.jnt_qposadr[_joint_id(mujoco, model, names)]) for names in ARM_JOINT_CANDIDATES]
    finger_ids = [
        index for index in range(model.njnt)
        if "/left/" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "").lower().replace(":", "/")
        and "finger" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "").lower()
    ]
    if not finger_ids:
        raise RuntimeError("找不到 ALOHA 左夹爪关节")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera_id < 0:
        raise RuntimeError(f"模型中没有相机: {args.camera}")

    try:
        import av
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 av；请用 venv Python 安装 av") from error
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        with av.open(str(output), "w") as container:
            stream = container.add_stream("libx264", rate=max(1, round(fps)))
            stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
            stream.time_base = Fraction(1, max(1, round(fps)))
            for index, (arm, gripper) in enumerate(zip(arm_states, gripper_states, strict=True)):
                data.qpos[arm_qpos] = arm
                for joint_id in finger_ids:
                    address = int(model.jnt_qposadr[joint_id])
                    low, high = model.jnt_range[joint_id]
                    data.qpos[address] = np.clip(gripper, low, high)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=args.camera)
                frame = av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    finally:
        renderer.close()
    print(f"已写出: {output}")


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"渲染失败: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
