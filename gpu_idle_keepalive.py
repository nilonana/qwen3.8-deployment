#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立的 Linux/vLLM 空闲负载脚本，仅依赖 Python 3 标准库。

python3 gpu_idle_keepalive.py probe   # 检查模型、指标和 GPU，不发推理请求
python3 gpu_idle_keepalive.py run     # 前台，Ctrl+C 停止
python3 gpu_idle_keepalive.py start   # 后台
python3 gpu_idle_keepalive.py stop
python3 gpu_idle_keepalive.py status

这是尽力而为的空闲负载控制，不保证零业务影响或最低利用率。
指标刷新及采样有延迟；取消连接后服务端取消也可能有延迟。
最多一条保活请求，不更改 vLLM、容器、进程名或业务客户端。
仅适合当前单引擎服务（tensor-parallel-size=1）。
修改下面配置后 stop/start 生效。API 密钥可用环境变量 VLLM_API_KEY。
"""

import argparse
import collections
import fcntl
import http.client
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

# ==================== 配置区 ====================
BASE_URL = "http://127.0.0.1:28000"
MODEL = "qwen3.8-27b"
GPU_DEVICE = "4"
TARGET_UTILIZATION = 55       # 调节目标，非利用率下限承诺
IDLE_SECONDS = 30.0          # 启动/检测到业务后，持续空闲多久才恢复
POLL_SECONDS = 1.0          # 业务指标轮询间隔；实际可见延迟还取决于服务端
GPU_POLL_SECONDS = 2.0
BURST_SECONDS = 10.0        # 每条保活请求最多保持多久，然后关闭连接
MIN_TOKENS = 128
MAX_TOKENS = 1024           # 空闲时逐步延长输出；不提高并发
METRICS_TIMEOUT = 2.0
DRAIN_SECONDS = 15.0        # 请求结束后等待服务端指标归零，期间不发新请求
ERROR_BACKOFF = 30.0
PROMPT = (
    "请持续用中文列举计算机视觉、三维几何和数值优化的基本知识，"
    "每项简短解释，尽可能写多项，不要提前总结。"
)
# =================================================

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "runs" / "idle_keepalive"
STATE_FILE = STATE_DIR / "process.json"
LOCK_FILE = STATE_DIR / "process.lock"
LOG_FILE = STATE_DIR / "idle_keepalive.log"
LOG = logging.getLogger("idle_keepalive")
STOP = threading.Event()


def setup_logging(background=False):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)
    if not background:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        LOG.addHandler(console)


def connection(timeout):
    parsed = urlsplit(BASE_URL)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("BASE_URL 必须是合法的 http/https 地址")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout)


def endpoint(path):
    return urlsplit(BASE_URL).path.rstrip("/") + path


def headers():
    result = {"Content-Type": "application/json", "User-Agent": "idle-keepalive/1.0"}
    key = os.environ.get("VLLM_API_KEY", "")
    if key:
        result["Authorization"] = "Bearer " + key
    return result


def get(path):
    conn = connection(METRICS_TIMEOUT)
    try:
        conn.request("GET", endpoint(path), headers=headers())
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path}: HTTP {response.status}, {body[:200]!r}")
        return body.decode("utf-8")
    finally:
        conn.close()


def parse_metrics(text):
    names = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    values = {name: [] for name in names}
    for line in text.splitlines():
        match = re.match(r"^(vllm:num_requests_(?:running|waiting))(?:\{[^}]*\})?\s+(\S+)", line)
        if match:
            value = float(match.group(2))
            if not math.isfinite(value) or value < 0:
                raise ValueError("请求数指标包含异常值")
            values[match.group(1)].append(value)
    if any(not values[name] for name in names):
        raise RuntimeError("/metrics 缺少 running/waiting 指标，暂停保活；检查是否关闭指标")
    # 对整个实例汇总，其他模型的业务也优先；不用于多引擎/DP 场景。
    return tuple(sum(values[name]) for name in names)


def metrics():
    return parse_metrics(get("/metrics"))


def gpu_utilization():
    result = subprocess.run([
        "nvidia-smi", "-i", GPU_DEVICE, "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=3)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi 失败")
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("GPU_DEVICE 必须指定单张 GPU")
    value = float(rows[0].strip())
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise RuntimeError("无法读取有效 GPU 利用率")
    return value


class LoadRequest:
    """单条可关闭 TCP 连接的流式请求。不使用后台无限重试。"""
    def __init__(self, tokens):
        self.tokens = tokens
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.sock = None
        self.error = None
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            if self.sock is not None:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.sock.close()

    def worker(self):
        conn = connection(METRICS_TIMEOUT)
        try:
            conn.connect()
            with self.lock:
                self.sock = conn.sock
                if self.cancelled.is_set():
                    return
            # 连续读取仍有整条请求的 BURST_SECONDS 限制，由控制循环关闭连接。
            conn.sock.settimeout(BURST_SECONDS + 5)
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": self.tokens,
                "temperature": 0,
                "stream": True,
                # vLLM 的扩展参数：只用于本脚本请求，确保输出不因 EOS 太早结束。
                "ignore_eos": True,
            }
            conn.request("POST", endpoint("/v1/chat/completions"),
                         body=json.dumps(payload).encode(), headers=headers())
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"推理 HTTP {response.status}: {response.read(300)!r}")
            saw_done = False
            while not self.cancelled.is_set():
                line = response.readline()
                if not line:
                    break
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        saw_done = True
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(f"流式错误: {chunk['error']}")
            if not saw_done and not self.cancelled.is_set():
                raise RuntimeError("流式响应提前断开，未收到 [DONE]")
        except Exception as exc:
            if not self.cancelled.is_set():
                self.error = str(exc)
        finally:
            conn.close()
            with self.lock:
                self.sock = None
            self.done.set()


def business_detected(running, waiting, own_active=False):
    # 自己最多一条；任何排队都主动让步（即使排队的恰好是自己）。
    return waiting > 0 or running > (1 if own_active else 0)


class Tuner:
    def __init__(self):
        self.tokens = MIN_TOKENS
        self.gap = 0.0
        self.samples = collections.deque(maxlen=5)
        self.last_adjustment = 0.0

    def observe(self, value, now, active):
        if not active:
            self.samples.clear()
            return
        self.samples.append(value)
        if len(self.samples) < 5 or now - self.last_adjustment < 10:
            return
        self.last_adjustment = now
        average = sum(self.samples) / len(self.samples)
        if average < TARGET_UTILIZATION - 5:
            self.gap = max(0.0, self.gap - 0.5)
            self.tokens = min(MAX_TOKENS, self.tokens * 2)
        elif average > TARGET_UTILIZATION + 10:
            self.gap = min(3.0, self.gap + 0.5)


def control_loop():
    tuner = Tuner()
    job = None
    drain_until = 0.0
    last_busy = time.monotonic()
    next_request = 0.0
    gpu_next = 0.0
    gpu_value = None
    last_summary = 0.0
    last_mode = None

    def mode(value):
        nonlocal last_mode
        if value != last_mode:
            LOG.info(value)
            last_mode = value

    LOG.info("启动: API=%s model=%s GPU=%s 目标=%s%% 空闲等待=%ss 最多1条保活请求",
             BASE_URL, MODEL, GPU_DEVICE, TARGET_UTILIZATION, IDLE_SECONDS)
    try:
        while not STOP.is_set():
            now = time.monotonic()
            if job and job.done.is_set():
                if job.error:
                    LOG.warning("保活请求失败: %s", job.error)
                    last_busy = now
                    next_request = now + ERROR_BACKOFF
                else:
                    next_request = now + tuner.gap
                job.thread.join(timeout=0.1)
                job = None
                drain_until = now + DRAIN_SECONDS

            try:
                running, waiting = metrics()
                if now >= gpu_next:
                    gpu_value = gpu_utilization()
                    gpu_next = time.monotonic() + GPU_POLL_SECONDS
            except (OSError, ValueError, RuntimeError, http.client.HTTPException,
                    subprocess.TimeoutExpired) as exc:
                if job:
                    job.cancel()
                last_busy = time.monotonic()
                gpu_value = None
                mode(f"监控不可用，暂停保活: {exc}")
                STOP.wait(max(POLL_SECONDS, 5))
                continue

            now = time.monotonic()
            own_active = job is not None
            # 请求结束后，历史指标可能尚未刷新；此时仅等待，不发新请求。
            draining = not own_active and running <= 1 and waiting == 0 and now < drain_until
            busy = business_detected(running, waiting, own_active)
            if busy and not draining:
                if job and not job.cancelled.is_set():
                    job.cancel()
                    LOG.info("检测到额外请求，关闭保活流连接，等待服务端处理取消")
                last_busy = now
                tuner.samples.clear()
                mode("业务繁忙，暂停保活")
            elif job:
                if now - job.started >= BURST_SECONDS:
                    job.cancel()
                mode("空闲保活中")
            elif running > 0 or waiting > 0:
                mode("等待服务端请求数归零，不发新请求")
            elif now - last_busy < IDLE_SECONDS:
                mode("等待持续空闲，再恢复保活")
            elif now >= next_request:
                job = LoadRequest(tuner.tokens)
                job.start()
                mode("空闲保活中")

            # 仅在已进入空闲负载阶段调节，不用业务产生的高利用率调节。
            tuner.observe(gpu_value, now, not busy and now - last_busy >= IDLE_SECONDS)
            if now - last_summary >= 60:
                LOG.info("状态: running=%s waiting=%s GPU=%s%% tokens=%s gap=%.1fs",
                         running, waiting, gpu_value, tuner.tokens, tuner.gap)
                if tuner.tokens == MAX_TOKENS and tuner.gap == 0 and gpu_value < TARGET_UTILIZATION:
                    LOG.info("已达单请求负载上限；当前硬件上可能无法达到利用率目标")
                last_summary = now
            STOP.wait(POLL_SECONDS)
    finally:
        if job:
            job.cancel()
            job.thread.join(timeout=METRICS_TIMEOUT + 1)
        LOG.info("已停止保活；vLLM 服务继续运行")


def process_start(pid):
    # /proc/stat 的 comm 可包含空格和括号，从最后一个右括号后解析。
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat.rsplit(")", 1)[1].split()[19]


def read_state():
    try:
        state = json.loads(STATE_FILE.read_text())
        pid = int(state["pid"])
        if process_start(pid) != state["start"]:
            return None
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(Path(__file__).resolve()).encode() not in cmdline or b"run" not in cmdline:
            return None
        return state
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def run(background=False):
    setup_logging(background)
    with LOCK_FILE.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有保活实例正在运行")
        pid = os.getpid()
        STATE_FILE.write_text(json.dumps({"pid": pid, "start": process_start(pid)}))
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        try:
            control_loop()
        finally:
            STATE_FILE.unlink(missing_ok=True)


def start():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if read_state():
        raise RuntimeError("已在运行，请先 stop")
    # 启动阶段错误单独保留，不和轮转日志竞争同一文件。
    with (STATE_DIR / "startup.log").open("ab") as output:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--background"],
            stdout=output, stderr=output, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=ROOT,
        )
    for _ in range(30):
        if proc.poll() is not None:
            raise RuntimeError(f"启动失败，查看 {STATE_DIR / 'startup.log'}")
        state = read_state()
        if state and state["pid"] == proc.pid:
            print(f"已后台启动 PID={proc.pid}；尚不代表 API 可用。日志: {LOG_FILE}")
            return
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("未能确认后台实例启动")


def stop():
    state = read_state()
    if not state:
        print("保活未运行")
        return
    pid = state["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(100):
        current = read_state()
        if not current or current != state:
            print("已停止保活；vLLM 服务继续运行")
            return
        time.sleep(0.1)
    raise RuntimeError("停止超时；未强制杀进程，请检查日志")


def probe():
    data = json.loads(get("/v1/models"))
    models = [item.get("id") for item in data.get("data", [])]
    if MODEL not in models:
        raise RuntimeError(f"未找到模型 {MODEL}，服务返回 {models}")
    running, waiting = metrics()
    util = gpu_utilization()
    print(f"检查通过: model={MODEL}, running={running}, waiting={waiting}, GPU={util}%")
    print("未发送推理请求。流式取消和业务延迟仍需在实际部署中验证。")


def status():
    state = read_state()
    print(f"运行中 PID={state['pid']}" if state else "保活未运行")
    if LOG_FILE.is_file():
        # 只读日志末尾 8 KiB。
        with LOG_FILE.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8192))
            for line in stream.read().decode("utf-8", errors="replace").splitlines()[-8:]:
                print(line)
    return 0 if state else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="run", choices=["run", "start", "stop", "status", "probe"])
    parser.add_argument("--background", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.action == "run":
            run(args.background)
        elif args.action == "start":
            start()
        elif args.action == "stop":
            stop()
        elif args.action == "probe":
            probe()
        else:
            return status()
        return 0
    except (OSError, ValueError, RuntimeError, http.client.HTTPException, subprocess.TimeoutExpired) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
