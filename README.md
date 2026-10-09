# 单张 RTX PRO 6000 部署 Qwen3.8-27B-FP8

更新时间：2026-10-09

使用 Linux 服务器、单张 NVIDIA RTX PRO 6000 Blackwell，通过 vLLM Docker 提供 OpenAI 兼容接口。

| 项目  | 本文配置 |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell，约 96 GB 显存，宿主机编号 4 |
| NVIDIA 驱动 | 590.48.01 |
| nvidia-smi 显示的 CUDA 版本 | 13.1 |
| 当前镜像 | docker.1ms.run/vllm/vllm-openai:v0.27.1 |
| 模型  | Qwen/Qwen3.8-27B-FP8 |
| 上下文上限 | 262144 tokens |
| 最大并发序列数 | 16  |
| 推测解码 | MTP，3 个推测 token |
| KV cache | FP8 |
| API 模型名 | qwen3.8-27b |
| 本机 Base URL | http://127.0.0.1:8000/v1 |

**验证状态：** 已验证 v0.27.1 的模型接口和实际回答正常；随后采用 FP8 KV cache、MTP 3 tokens、256K 上下文、8 并发配置，实际体验速度明显提升，但未记录量化测速结果。本文保留 v0.27.1 镜像，将并发上限改为 16，启动及访问端口统一为 8000。16 并发尚待验证；可选镜像升级另列在第三板块。

## 一、安装

### 1. 检查现有环境

本次服务器已经安装 Docker、NVIDIA 驱动，并具备容器 GPU 支持。本文记录基于现有环境的部署，不包含这些组件从零安装的过程。

```bash
docker --version
docker info
docker image ls
docker ps -a
nvidia-smi
```

本文使用宿主机 4 号卡；其他服务器应按实际编号调整。nvidia-smi 的 CUDA Version 表示驱动支持的版本，不代表宿主机安装了相同版本的 CUDA Toolkit。

### 2. 准备目录和下载环境

使用当前用户目录，避免在 GitHub 记录中写入私人用户名：

```bash
mkdir -p "$HOME/WorkStation/Vllm/models"

python3 -m venv "$HOME/WorkStation/Vllm/download-env"
"$HOME/WorkStation/Vllm/download-env/bin/pip" install -U huggingface_hub
```

如果 pip 也需要代理：

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
ALL_PROXY=http://127.0.0.1:7897 \
"$HOME/WorkStation/Vllm/download-env/bin/pip" install -U huggingface_hub
```

宿主机只安装下载工具，vLLM 在容器内运行，无需修改现有算法开发环境。

### 3. 拉取镜像并验证 GPU

```bash
docker pull docker.1ms.run/vllm/vllm-openai:v0.27.1

docker run --rm \
  --gpus '"device=4"' \
  --entrypoint nvidia-smi \
  docker.1ms.run/vllm/vllm-openai:v0.27.1
```

容器内应只看到目标卡，编号可能重新排列为 0。这里使用本次实际部署的 v0.27.1 镜像，本地已有镜像时可跳过拉取。

Docker 镜像包含运行环境，不包含需要单独下载的模型权重。docker pull 不会启动服务，也不会更新已有容器。

### 4. 通过代理下载模型

本次成功使用服务器本机 HTTP 代理，端口为 7897。

```bash
ss -lntp 'sport = :7897'

curl -I --connect-timeout 10 --max-time 30 \
  --proxy http://127.0.0.1:7897 \
  https://huggingface.co
```

下载完整仓库：

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
ALL_PROXY=http://127.0.0.1:7897 \
"$HOME/WorkStation/Vllm/download-env/bin/hf" download \
  Qwen/Qwen3.8-27B-FP8 \
  --local-dir "$HOME/WorkStation/Vllm/models/Qwen38-27B-FP8"
```

等待命令成功结束；中断后重新执行同一命令可复用已下载内容。代理变量仅作用于本次进程。如果模型已完整下载，可跳过本步骤。

检查文件：

```bash
ls -lh "$HOME/WorkStation/Vllm/models/Qwen38-27B-FP8/config.json"
du -sh "$HOME/WorkStation/Vllm/models/Qwen38-27B-FP8"
```

文件存在和目录大小只是初步检查；完整性最终需要结合下载成功、模型加载和实际生成确认。

## 二、启动和停止

### 1. 首次创建：当前镜像、16 并发

先查看旧服务，停止同一 GPU 上实际运行的推理容器：

```bash
docker ps
```

例如，若当前运行的是已验证提速的容器：

```bash
docker stop qwen38-27b-mtp
```

如果运行的是最初的 qwen38-27b-vllm 或其他容器，应替换成实际名称。16 并发使用新的名称 qwen38-27b-mtp16，保留原来的 8 并发容器供回退。

```bash
docker run -d \
  --name qwen38-27b-mtp16 \
  --gpus '"device=4"' \
  --ipc=host \
  --restart=no \
  -p 127.0.0.1:28000:8000 \
  -v "$HOME/WorkStation/Vllm/models:/models:ro" \
  -e HF_HUB_OFFLINE=1 \
  docker.1ms.run/vllm/vllm-openai:v0.27.1 \
  /models/Qwen38-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \ # 改成16384试试
  --max-num-seqs 16 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

这里保留当前 v0.27.1，不进行镜像升级。此前实测提速的是 8 并发配置，改为 16 后需再次验证。

| 参数  | 说明  |
| --- | --- |
| --gpus device=4 | 只暴露宿主机 4 号卡 |
| --tensor-parallel-size 1 | 单卡推理 |
| --restart=no | 不自动重启 |
| -p 127.0.0.1:8000:8000 | 服务器本机 8000 映射至容器 8000 |
| -v ...:/models:ro | 只读挂载模型目录 |
| HF_HUB_OFFLINE=1 | 使用本地文件，禁止 Hugging Face Hub 在线访问 |
| --dtype bfloat16 | 设置计算及非量化部分的数据类型，权重量化按模型配置读取 |
| --max-model-len 262144 | 单条序列输入和输出合计上限 |
| --max-num-batched-tokens 8192 | 每次调度的 token 预算，不是上下文上限 |
| --max-num-seqs 16 | 同时处理的最大序列数 |
| --kv-cache-dtype fp8 | 降低 KV cache 显存占用 |
| --enable-prefix-caching | 复用相同前缀 |
| --enable-chunked-prefill | 分块调度长输入预填充 |
| --reasoning-parser qwen3 | 解析思考与正文，不会关闭思考 |
| --speculative-config | 启用 MTP，推测 3 个 token |

16 并发不代表 16 条最大长度请求一定能同时驻留显存。不强制指定量化和内核后端，使用权重自带量化配置及框架默认优化。

### 2. 查看日志并验证

```bash
docker logs -f --tail 150 qwen38-27b-mtp16
```

首次加载和编译可能需要数分钟。Ctrl+C 只退出日志查看，不会停止容器。

```bash
curl http://127.0.0.1:8000/v1/models
```

再测试实际回答：

```bash
curl --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

此处关闭思考用于快速验证，需要推理能力时可按任务开启。

### 3. 停止、再启动和重启

```bash
# 停止服务
docker stop qwen38-27b-mtp16

# 查看状态，停止后应为 Exited
docker ps -a --filter name=qwen38-27b-mtp16

# 使用原配置再次启动
docker start qwen38-27b-mtp16

# 重启
docker restart qwen38-27b-mtp16
```

停止会释放推理进程的 GPU 占用，保留容器和模型文件。docker start 不修改参数；改镜像或并发数需要创建新容器。

### 4. Windows 客户端访问

宿主机端口只绑定本机，通过 SSH 转发访问。在 Windows PowerShell 执行，替换用户名与服务器地址：

```powershell
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER_IP
```

保持窗口打开，客户端填写：

| 配置  | 值   |
| --- | --- |
| Base URL | http://127.0.0.1:8000/v1 |
| Model | qwen3.8-27b |
| API Key | 客户端必填时可填 EMPTY |

本文未设置服务端 API Key，EMPTY 只是客户端占位值。本机端口被占用时，更换 SSH 左侧端口并同步修改客户端地址。

## 三、更新

### 1. 可选更新目标与依据（当前尚未执行）

更新的是 vLLM 镜像，不是 Docker 引擎。Docker 已正常访问 GPU，暂无依据要求升级引擎。现有模型权重无需因为镜像更新而重新下载。

截至本文日期：

- GitHub 最新正式版本为 v0.31.0，官方默认镜像使用 CUDA 13.0。
- v0.29.0 发布记录包含 Qwen GDN/MTP 融合内核、推测解码 CUDA Graph 与缓存优化。
- Qwen3.8 官方配方提供 MTP 3 tokens、FP8 KV cache。另有 NVFP4 低延迟权重路线，需要单独下载与验证。

因此值得更新测试，但不能保证每台机器同样提速。16 并发及 8192 调度预算是本地选择，不是官方统一最佳参数。

### 2. 更新流程

1. 拉取新镜像。
2. 停止实际运行的旧服务。
3. 复制第二板块的启动命令，将镜像替换为 vllm/vllm-openai:v0.31.0，容器名改为 qwen38-v031-mtp16；其余参数和端口 8000 保持一致。
4. 验证日志、回答、速度及显存。
5. 新版确认满足需求后，再考虑清理旧版本。

```bash
docker pull vllm/vllm-openai:v0.31.0
docker ps -a
```

新镜像拉取成功后，停止当前容器，再按上述替换创建新版：

```bash
docker stop qwen38-27b-mtp16
```

若实际运行的仍是原来的 8 并发容器，则改为停止 qwen38-27b-mtp。

只拉取镜像不会更新已有容器；docker restart 也不会更换容器镜像。

### 3. 回退

如果新版运行不理想：

```bash
docker stop qwen38-v031-mtp16
docker start qwen38-27b-mtp16
```

以上恢复第二板块创建的当前版 16 并发容器；要恢复已验证的 8 并发版本，可启动 qwen38-27b-mtp。容器名称以 docker ps -a 为准。

### 4. 固定请求测速

在服务器本机测试。保持提示词、输出长度、思考开关及并发负载一致，第一次请求用于热身。

以下是三次串行测试，速率包含输入处理时间，不是纯解码速度，也不能验证 16 并发吞吐：

```bash
python3 - <<'PY'
import json
import time
import urllib.request

payload = {
    "model": "qwen3.8-27b",
    "messages": [{
        "role": "user",
        "content": "请详细介绍激光雷达与相机融合的原理、主要方法和应用场景。"
    }],
    "max_tokens": 512,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": False}
}
for i in range(3):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    tokens = result["usage"]["completion_tokens"]
    print(
        f"第{i+1}次: {elapsed:.2f}秒, 输出{tokens} tokens, "
        f"端到端速率{tokens/elapsed:.2f} tokens/s"
    )
PY
```

重复请求可能命中前缀缓存；比较时使用相同流程。按实际需求补充流式回复、长对话和工具调用验证。

参考：

- [官方模型仓库](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Qwen3.8 vLLM 配方](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM v0.31.0 发布记录](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
- [vLLM v0.29.0 发布记录](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM Docker 文档](https://docs.vllm.ai/en/latest/deployment/docker/)

## 四、可能遇到的问题

### 1. Exited (1)：模型目录为空或路径错误

本次最初失败的原因是宿主机挂载目录为空，日志为：

```text
Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/models/Qwen38-27B-FP8'
```

检查实际容器，替换 CONTAINER_NAME：

```bash
docker logs --tail 200 CONTAINER_NAME
docker inspect CONTAINER_NAME --format '{{json .Config.Cmd}}'
docker inspect CONTAINER_NAME --format '{{range .Mounts}}{{println .Source " -> " .Destination}}{{end}}'
```

| 宿主机路径 | 容器内路径 |
| --- | --- |
| $HOME/WorkStation/Vllm/models | /models |
| $HOME/WorkStation/Vllm/models/Qwen38-27B-FP8 | /models/Qwen38-27B-FP8 |

下载到真实挂载目录，不能只拉取镜像。该错误发生在配置读取阶段，不能解释为显存不足。

### 2. 模型下载 Network is unreachable

本次通过显式设置三项代理变量解决，见第一板块。检查代理监听和 curl 连接；如果代理在另一台电脑，服务器的 127.0.0.1 不会自动指向它。

### 3. Docker 拉取失败，但 hf download 成功

Docker daemon 的联网配置与下载进程不同。给 hf download 或 docker pull 添加普通终端代理变量，不等于配置 daemon 代理。

按服务器管理方式配置 [Docker daemon 代理](https://docs.docker.com/engine/daemon/proxy/)。共享服务器上的 daemon 配置和重启应按相应管理流程处理。

也可在联网电脑上拉取并导出镜像：

```bash
docker pull --platform linux/amd64 vllm/vllm-openai:v0.27.1
docker save -o vllm-v0271.tar vllm/vllm-openai:v0.27.1
```

linux/amd64 以目标服务器为 x86_64 为前提，用 uname -m 确认；其他架构应调整。传到服务器后：

```bash
docker load -i vllm-v0271.tar
```

### 4. Windows 下载证书错误

曾遇到 CERTIFICATE_VERIFY_FAILED / self signed certificate in certificate chain，Windows 路线未完成验证。最终使用服务器代理下载成功。若继续 Windows 下载，应排查代理证书与 Python 信任库。

### 5. 容器名称或端口冲突

```bash
docker ps -a --filter name=qwen38-27b-mtp16
ss -lntp 'sport = :8000'
```

已有正确配置的停止容器可用 docker start。改参数时另建新名称，保留旧容器。端口冲突时停止旧服务或更换宿主机端口，同步修改 curl、测速、SSH 和客户端。

### 6. GPU 不可用或显存不足

```bash
nvidia-smi
docker logs --tail 200 qwen38-27b-mtp16
```

确认 GPU 编号及旧服务是否停止。256K 是每条序列上限，不保证 16 条最大长度请求都能驻留。实际长输入和高并发可能排队或增加缓存压力。

根据日志确定错误阶段再调整容量参数，不要未经判断关闭 CUDA Graph。

### 7. GPU 占用存在，但接口不可用

显存占用不代表服务已经就绪。查看日志，分别验证模型列表和实际生成。本文宿主机和容器端口均为 8000。外部电脑需通过 SSH 转发访问。

### 8. 响应慢

非流式请求等整段生成结束才返回；思考模式也可能产生大量推理内容。reasoning-parser 不会关闭思考。MTP 收益需实测，并发增大不保证单请求更快。

256K 上限不会让短请求自动变成 256K 输入；实际长输入仍有额外成本。使用第三板块测试，区分等待体感和实际生成速度。

### 9. GitHub 中不要提交权重与环境

本文可改名为 README.md。若仓库位于部署目录内，按需要加入 .gitignore：

```gitignore
models/
download-env/
*.tar
```

仅提交说明和必要配置，不提交模型权重、虚拟环境、个人登录信息。
