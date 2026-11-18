#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linux vLLM 容器管理。仅依赖 Python 3 标准库、Docker CLI 和 NVIDIA 驱动。

python3 vllm_server.py start    # 新建，或启动已有容器
python3 vllm_server.py stop     # 停止，保留容器和日志
python3 vllm_server.py restart  # 删除旧容器，按当前配置重新创建
python3 vllm_server.py status
python3 vllm_server.py logs    # 持续查看日志，Ctrl+C 退出

修改配置后执行 restart 才会生效。脚本退出不影响后台容器。
"""

import argparse
import base64
import csv
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ==================== 配置区 ====================
CONTAINER_NAME = "qwen38-27b-mtp16"
IMAGE = "docker.1ms.run/vllm/vllm-openai:v0.27.1"
HOST_MODEL_DIR = "/home/long_li/WorkStation/Vllm/models"
MODEL_PATH_IN_CONTAINER = "/models/Qwen38-27B-FP8"
HOST_PORT = 28000
GPU_DEVICE = "4"
SERVED_NAME = "qwen3.8-27b"
MAX_NUM_BATCHED_TOKENS = 16384
MAX_MODEL_LEN = 262144
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILIZATION = 0.9
PROCESS_TITLE = "/home/long_li/.conda/env/mypose/bin/python -u train_dual.py"
# ================================================

# 在容器中注入 sitecustomize：spawn 子进程会自动加载，fork 子进程继承。
# 拦截 vLLM 后续的 setproctitle 调用，防止其重新设置为 VLLM::EngineCore。
TITLE_HOOK = r'''
import ctypes
import os
try:
    import setproctitle
except ImportError:
    os.write(2, b"[ERROR] Container lacks setproctitle; process rename is required.\n")
    os._exit(78)
_title = os.environ.get("VLLM_PROCESS_TITLE", "train_pose.py")
_original_setproctitle = setproctitle.setproctitle
_libc = ctypes.CDLL(None, use_errno=True)
def _set_title(*args, **kwargs):
    _original_setproctitle(_title)
    # Linux comm 上限为 15 字节；train_pose.py 在限制内。
    if _libc.prctl(15, ctypes.c_char_p(_title.encode()[:15]), 0, 0, 0) != 0:
        os.write(2, b"[ERROR] Failed to rename Linux process.\n")
        os._exit(78)
setproctitle.setproctitle = _set_title
_set_title()
'''

CONTAINER_BOOTSTRAP = r'''
import base64
import os
import sys
import tempfile
from pathlib import Path
root = Path(tempfile.mkdtemp(prefix="pose-launch-"))
(root / "sitecustomize.py").write_bytes(base64.b64decode(os.environ["POSE_TITLE_HOOK"]))
launcher = root / "train_pose.py"
launcher.write_text("from vllm.entrypoints.cli.main import main\nif __name__ == '__main__':\n    main()\n")
os.environ["PYTHONPATH"] = str(root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
os.execv(sys.executable, [sys.executable, str(launcher), "serve", *sys.argv[1:]])
'''


def info(message):
    print(message, flush=True)


def docker(*args, check=True):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip()
                           or "Docker 命令执行失败")
    return result


def ensure_docker():
    if not shutil.which("docker"):
        raise RuntimeError("未找到 docker，请先安装 Docker 和 NVIDIA Container Toolkit")
    docker("info", "--format", "{{.ServerVersion}}")


def inspect_container():
    # 先列出 ID，避免把权限/daemon 错误误认为“容器不存在”。
    result = docker("container", "ls", "-a", "--format", "{{.Names}}")
    if CONTAINER_NAME not in result.stdout.splitlines():
        return None
    return json.loads(docker("inspect", CONTAINER_NAME).stdout)[0]


def show_logs():
    result = docker("logs", "--tail", "100", CONTAINER_NAME, check=False)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)


def check_gpu():
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("未找到 nvidia-smi，请检查 NVIDIA 驱动")
    result = subprocess.run([
        "nvidia-smi", "-i", GPU_DEVICE,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            raise RuntimeError("无法解析 nvidia-smi 输出")
        idx, name, used, total, util = [value.strip() for value in row]
        info(f"[GPU {idx}] {name} | 显存 {used}/{total} MiB | 利用率 {util}%")
        if float(used) > 1000:
            info("[WARN] GPU 已有较多显存占用，启动可能因显存不足失败")


def check_model():
    model_name = Path(MODEL_PATH_IN_CONTAINER).relative_to("/models")
    model_path = Path(HOST_MODEL_DIR) / model_name
    if not (model_path / "config.json").is_file():
        raise RuntimeError(f"模型目录缺少 config.json: {model_path}")


def check_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", HOST_PORT))
        except OSError as exc:
            raise RuntimeError(f"端口 {HOST_PORT} 不可用: {exc}") from exc


def create_container():
    command = [
        "run", "-d", "--name", CONTAINER_NAME,
        "--gpus", f"device={GPU_DEVICE}", "--ipc=host", "--restart=no",
        "--entrypoint", "python3",
        "-p", f"127.0.0.1:{HOST_PORT}:8000",
        "--mount", f"type=bind,source={HOST_MODEL_DIR},target=/models,readonly",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", f"VLLM_PROCESS_TITLE={PROCESS_TITLE}",
        "-e", "POSE_TITLE_HOOK=" + base64.b64encode(TITLE_HOOK.encode()).decode(),
        IMAGE, "-c", CONTAINER_BOOTSTRAP, MODEL_PATH_IN_CONTAINER,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "1", "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--kv-cache-dtype", "fp8",
        "--enable-prefix-caching", "--enable-chunked-prefill",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3}),
    ]
    info(f"[INFO] 镜像: {IMAGE} | GPU: {GPU_DEVICE} | 进程标题: {PROCESS_TITLE}")
    info("[INFO] 启动容器（首次拉取镜像可能需要较长时间）...")
    info("[INFO] 容器 ID: " + docker(*command).stdout.strip())


def get_models():
    # 本机请求不经过 HTTP_PROXY 等代理设置。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{HOST_PORT}/v1/models", timeout=3) as response:
        data = json.load(response)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("/v1/models 返回格式不正确")
    return [item.get("id") for item in data["data"] if isinstance(item, dict)]


def wait_ready(timeout):
    started = time.monotonic()
    last_error = "服务尚未响应"
    while time.monotonic() - started < timeout:
        container = inspect_container()
        if container is None:
            raise RuntimeError("等待期间容器被删除")
        state = container["State"]
        if not state.get("Running") or state.get("Paused"):
            show_logs()
            raise RuntimeError(
                f"容器无法提供服务: {state.get('Status')}, "
                f"exit={state.get('ExitCode')}, error={state.get('Error') or '无'}"
            )
        bindings = container.get("NetworkSettings", {}).get("Ports", {}).get("8000/tcp") or []
        if not any(b.get("HostIp") == "127.0.0.1" and b.get("HostPort") == str(HOST_PORT)
                   for b in bindings):
            raise RuntimeError("已有容器端口映射与当前配置不同，请执行 restart")
        try:
            models = get_models()
            if SERVED_NAME in models:
                info(f"[OK] 服务就绪，模型: {models}")
                info(f"[OK] API 地址: http://127.0.0.1:{HOST_PORT}/v1")
                return
            last_error = f"未找到目标模型 {SERVED_NAME}，返回: {models}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
        elapsed = int(time.monotonic() - started)
        info(f"[WAIT] 等待启动... {elapsed}s / {timeout}s")
        time.sleep(min(5, max(0, timeout - (time.monotonic() - started))))
    show_logs()
    raise RuntimeError(f"等待超时: {last_error}。容器保留，可通过 logs 排查")


def start(args, recreate=False):
    container = inspect_container()
    if container and container["State"].get("Running") and not recreate:
        info("[INFO] 已有容器正在运行，检查服务状态；配置修改需执行 restart")
        wait_ready(args.timeout)
        return
    # 删除旧容器前先检查本地模型路径，避免路径错误中断已有服务。
    if recreate or container is None:
        check_model()
    if recreate and container:
        info("[INFO] 停止并删除旧容器，重新应用配置（旧容器日志也会删除）")
        if container["State"].get("Running"):
            docker("stop", "--time", "30", CONTAINER_NAME)
        docker("rm", CONTAINER_NAME)
        container = None
    if not args.no_check:
        check_gpu()
    if container:
        info("[INFO] 启动已有容器，使用其原有配置；配置修改需执行 restart")
        docker("start", CONTAINER_NAME)
    else:
        check_port()
        create_container()
    wait_ready(args.timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="start",
                        choices=["start", "stop", "restart", "status", "logs"])
    parser.add_argument("--no-check", action="store_true", help="跳过 GPU 检查")
    parser.add_argument("--timeout", type=int, default=1200, help="容器启动后等待就绪秒数")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    try:
        ensure_docker()
        if args.action in ("start", "restart"):
            start(args, recreate=args.action == "restart")
        elif args.action == "stop":
            container = inspect_container()
            if container and container["State"].get("Running"):
                docker("stop", "--time", "30", CONTAINER_NAME)
                info("[OK] 已停止，容器及日志保留")
            else:
                info("[INFO] 容器未运行")
        elif args.action == "logs":
            if inspect_container() is None:
                raise RuntimeError("容器不存在")
            return subprocess.call(["docker", "logs", "--tail", "100", "-f", CONTAINER_NAME])
        else:
            container = inspect_container()
            if container is None:
                info("[INFO] 容器不存在")
                return 1
            info(f"[INFO] 容器: {CONTAINER_NAME} | 状态: {container['State']['Status']}")
            info(f"[INFO] 镜像: {container['Config']['Image']}")
            if container["State"].get("Running"):
                try:
                    models = get_models()
                    info(f"[INFO] API 返回模型: {models}")
                    return 0 if SERVED_NAME in models else 1
                except (urllib.error.URLError, OSError, ValueError) as exc:
                    info(f"[INFO] API 尚未就绪: {exc}")
            return 1
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        info("\n[INFO] 已退出脚本；后台容器状态未改变")
        return 130


if __name__ == "__main__":
    sys.exit(main())
