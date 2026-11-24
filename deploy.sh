#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI_CIX_HOME="${PI_CIX_HOME:-$HOME/.local/share/pi-agent-cix}"
ENV_FILE="$PI_CIX_HOME/config.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

PI_CIX_HOME="${PI_CIX_HOME:-$HOME/.local/share/pi-agent-cix}"
MODEL_ROOT="${MODEL_ROOT:-$PI_CIX_HOME/models}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-$MODEL_ROOT/onnx_sensevoice}"
TTS_MODEL_DIR="${TTS_MODEL_DIR:-$MODEL_ROOT/onnx_kokoro}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_ROOT/Qwen3.5-4B-MNN}"
MNN_SOURCE_DIR="${MNN_SOURCE_DIR:-$PI_CIX_HOME/source/MNN}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-$MNN_SOURCE_DIR/build-cix-opencl}"
MNNCLI_BUILD_DIR="${MNNCLI_BUILD_DIR:-$MNN_SOURCE_DIR/apps/mnncli/build_mnncli}"
MNNCLI="$MNNCLI_BUILD_DIR/mnncli"
OPENCL_RUNTIME_LIB="${OPENCL_RUNTIME_LIB:-/opt/cixgpu-pro/lib/aarch64-linux-gnu}"
CIX_WHEEL_DIR="${CIX_WHEEL_DIR:-/usr/share/cix/pypi}"
CIX_MODEL_REPO="${CIX_MODEL_REPO:-cix/ai_model_hub_25_Q3}"
QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-MNN/Qwen3.5-4B-MNN}"
MNN_API_HOST="${MNN_API_HOST:-127.0.0.1}"
MNN_BACKEND_PORT="${MNN_BACKEND_PORT:-8001}"
MNN_PROXY_PORT="${MNN_PROXY_PORT:-8000}"
PI_PROVIDER="${PI_PROVIDER:-mnn}"
PI_MODEL="${PI_MODEL:-Qwen3.5-4B-MNN}"
PI_TOOLS="${PI_TOOLS:-read,grep,find,ls}"
PI_WORKSPACE_DIR="${PI_WORKSPACE_DIR:-$PI_CIX_HOME/workspace}"
PI_SESSION_FILE="${PI_SESSION_FILE:-$PI_CIX_HOME/sessions/voice.jsonl}"
PI_AGENT_DIR="$PI_CIX_HOME/pi-agent"
NODE_HOME="$PI_CIX_HOME/node"
PI_CLI="$NODE_HOME/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
PYTHON="$PI_CIX_HOME/venv/bin/python"
LOG_DIR="$PI_CIX_HOME/logs"
PID_DIR="$PI_CIX_HOME/run"
MNN_COMMIT="dbf9ba61b08106a3b43426e295d1a29cb41cc8fe"
JOBS="${JOBS:-4}"

die() { printf '错误：%s\n' "$*" >&2; exit 1; }

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

check_board() {
  [[ "$(uname -m)" == "aarch64" ]] || die "需要 AArch64 CIX 开发板。"
  compgen -G "/dev/aipu*" >/dev/null || die "找不到 /dev/aipu*；请先安装 CIX 官方系统及 NPU 驱动。"
  [[ -e /dev/mali0 ]] || die "找不到 /dev/mali0，MNN OpenCL 后端不可用。"
  [[ -d "$OPENCL_RUNTIME_LIB" ]] || die "找不到 CIX OpenCL 库目录：$OPENCL_RUNTIME_LIB"
  [[ -d /usr/share/cix/lib/onnxruntime/operator ]] || die "找不到 CIX NPU operator 目录，请按板卡手册安装运行库。"
}

load_runtime_env() {
  export ASR_MODEL_DIR TTS_MODEL_DIR
  export PI_CODING_AGENT_DIR="$PI_AGENT_DIR"
  export PI_CODING_AGENT_SESSION_DIR="$PI_CIX_HOME/sessions"
  export PI_WORKSPACE_DIR PI_SESSION_FILE PI_PROVIDER PI_MODEL PI_TOOLS PI_CLI
  export LLM_MODEL="$PI_MODEL" LLM_BASE_URL="http://$MNN_API_HOST:$MNN_PROXY_PORT/v1"
  export MNN_BACKEND_URL="http://$MNN_API_HOST:$MNN_BACKEND_PORT"
  export MNN_PROXY_HOST="$MNN_API_HOST" MNN_PROXY_PORT
  export LD_LIBRARY_PATH="/usr/share/cix/lib/onnxruntime:$OPENCL_RUNTIME_LIB:$MNN_BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export OPERATOR_PATH="/usr/share/cix/lib/onnxruntime/operator"
}

install_system_packages() {
  as_root apt-get update
  as_root apt-get install -y build-essential cmake git git-lfs pkg-config libssl-dev \
    libcurl4-openssl-dev libopencl-dev ocl-icd-opencl-dev python3 python3-pip \
    python3-venv curl ca-certificates alsa-utils libasound2-dev libsndfile1 espeak-ng
}

ensure_python311() {
  mkdir -p "$PI_CIX_HOME"
  if [[ -x "$PI_CIX_HOME/venv/bin/python" ]] && "$PI_CIX_HOME/venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,11)' >/dev/null 2>&1; then return; fi
  rm -rf "$PI_CIX_HOME/venv"
  if command -v python3.11 >/dev/null 2>&1 && python3.11 -m venv "$PI_CIX_HOME/venv"; then return; fi
  rm -rf "$PI_CIX_HOME/venv"
  local installer="$PI_CIX_HOME/Miniforge3-Linux-aarch64.sh"
  if [[ ! -x "$PI_CIX_HOME/miniforge3/bin/conda" ]]; then
    curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -o "$installer"
    bash "$installer" -b -p "$PI_CIX_HOME/miniforge3"
  fi
  "$PI_CIX_HOME/miniforge3/bin/conda" create -y -p "$PI_CIX_HOME/venv" python=3.11 pip
}

install_python_packages() {
  ensure_python311
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
  [[ -d "$CIX_WHEEL_DIR" ]] || die "找不到 CIX 官方 wheel 目录：$CIX_WHEEL_DIR"
  "$PYTHON" - "$CIX_WHEEL_DIR" <<'PY'
import glob, pathlib, subprocess, sys
base = pathlib.Path(sys.argv[1])
patterns = ["libnoe-*-aarch64.whl", "NOE_Engine-*.whl",
            "onnxruntime_zhouyi-*-cp311-*-aarch64.whl", "ZhouyiOperators_x2-*.whl"]
wheels = []
for pattern in patterns:
    matches = sorted(glob.glob(str(base / pattern)))
    if not matches: raise SystemExit(f"缺少 CIX 官方 wheel: {pattern}")
    wheels.append(matches[-1])
subprocess.check_call([sys.executable, "-m", "pip", "install", *wheels])
PY
}

download_models() {
  mkdir -p "$MODEL_ROOT" "$QWEN_MODEL_DIR"
  local hub_root="$MODEL_ROOT/cix_model_hub"
  "$PYTHON" - "$CIX_MODEL_REPO" "$hub_root" "$QWEN_MODEL_REPO" "$QWEN_MODEL_DIR" <<'PY'
import sys
from pathlib import Path
from modelscope import snapshot_download
cix_repo, hub_root, qwen_repo, qwen_root = sys.argv[1:]
hub_root, qwen_root = Path(hub_root), Path(qwen_root)
asr = "models/Audio/Speech_Recognition/onnx_sensevoice/"
tts = "models/Audio/TTS/onnx_kokoro/"
patterns = [
    asr + "sense_voice_mask256.cix", asr + "model/am.mvn", asr + "model/config.yaml",
    asr + "model/chn_jpn_yue_eng_ko_spectok.bpe.model", asr + "model/LICENSE",
    asr + "model/MODEL_LICENSE", asr + "funasr_onnx/**",
    tts + "bert2.cix", tts + "predictor.cix", tts + "text_encoder.cix",
    tts + "decoder00.cix", tts + "decoder01.cix", tts + "decoder1_int8.mnn",
    tts + "datasets/npu_config.json", tts + "voices/zf_xiaoyi.pt",
    tts + "voices/af_heart.pt", tts + "kokoro/**", tts + "model/LICENSE",
]
snapshot_download(repo_id=cix_repo, local_dir=str(hub_root), allow_file_pattern=patterns)
for stable, relative in [
    (Path(sys.argv[2]).parent / "onnx_sensevoice", Path(asr)),
    (Path(sys.argv[2]).parent / "onnx_kokoro", Path(tts)),
]:
    source = hub_root / relative
    if not source.exists(): raise SystemExit(f"模型文件未下载：{source}")
    if stable.is_symlink() and stable.resolve() == source.resolve(): continue
    if stable.is_dir() and not any(stable.iterdir()): stable.rmdir()
    if not stable.exists(): stable.symlink_to(source, target_is_directory=True)
snapshot_download(
    repo_id=qwen_repo, local_dir=str(qwen_root),
    allow_file_pattern=["config-opencl.json", "config.json", "configuration.json",
                        "llm_config.json", "llm.mnn", "llm.mnn.json",
                        "llm.mnn.weight", "tokenizer.txt", "README.md", "*.license"],
)
PY
  [[ -f "$ASR_MODEL_DIR/sense_voice_mask256.cix" ]] || die "SenseVoice 权重未下载到 $ASR_MODEL_DIR"
  [[ -f "$TTS_MODEL_DIR/datasets/npu_config.json" ]] || die "Kokoro 配置未下载到 $TTS_MODEL_DIR"
  [[ -f "$QWEN_MODEL_DIR/config-opencl.json" ]] || die "Qwen3.5-4B OpenCL 配置未下载到 $QWEN_MODEL_DIR"
}

build_mnn() {
  mkdir -p "$(dirname -- "$MNN_SOURCE_DIR")"
  if [[ ! -d "$MNN_SOURCE_DIR/.git" ]]; then
    git clone --recursive https://github.com/alibaba/MNN.git "$MNN_SOURCE_DIR"
  fi
  git -C "$MNN_SOURCE_DIR" fetch origin
  git -C "$MNN_SOURCE_DIR" checkout --detach "$MNN_COMMIT"
  git -C "$MNN_SOURCE_DIR" submodule update --init --recursive
  if git -C "$MNN_SOURCE_DIR" apply --check "$SCRIPT_DIR/patches/mnncli-cix-openai.patch"; then
    git -C "$MNN_SOURCE_DIR" apply "$SCRIPT_DIR/patches/mnncli-cix-openai.patch"
  elif ! git -C "$MNN_SOURCE_DIR" apply --reverse --check "$SCRIPT_DIR/patches/mnncli-cix-openai.patch"; then
    die "MNN CLI 补丁和源码不匹配；检查固定版本与 patches 目录。"
  fi

  local opencl_library
  opencl_library="$(ldconfig -p 2>/dev/null | awk '/libOpenCL\.so / {print $NF; exit}')"
  [[ -n "$opencl_library" ]] || die "找不到 libOpenCL.so"
  cmake -S "$MNN_SOURCE_DIR" -B "$MNN_BUILD_DIR" \
    -DMNN_BUILD_LLM=ON -DMNN_OPENCL=ON -DMNN_LOW_MEMORY=ON \
    -DMNN_BUILD_SHARED_LIBS=ON -DMNN_KLEIDIAI=ON \
    -DMNN_KLEIDIAI_DEFAULT_ON=ON -DMNN_USE_SYSTEM_LIB=ON \
    -DOpenCL_INCLUDE_DIR=/usr/include -DOpenCL_LIBRARY="$opencl_library"
  cmake --build "$MNN_BUILD_DIR" --parallel "$JOBS"
  cmake -S "$MNN_SOURCE_DIR/apps/mnncli" -B "$MNNCLI_BUILD_DIR" \
    -DMNN_BUILD_DIR="$MNN_BUILD_DIR" -DMNN_SOURCE_DIR="$MNN_SOURCE_DIR"
  cmake --build "$MNNCLI_BUILD_DIR" --target mnncli --parallel "$JOBS"
  [[ -x "$MNNCLI" ]] || die "mnncli 编译失败"
}

install_node_and_pi() {
  local version major minor setup
  version="$(node --version 2>/dev/null || true)"
  major="$(printf '%s' "$version" | sed -E 's/^v([0-9]+).*/\1/')"
  minor="$(printf '%s' "$version" | sed -E 's/^v[0-9]+\.([0-9]+).*/\1/')"
  if [[ -z "$major" ]] || (( major < 22 || (major == 22 && minor < 19) )); then
    setup="$PI_CIX_HOME/nodesource-setup.sh"
    curl -fsSL https://deb.nodesource.com/setup_24.x -o "$setup"
    as_root bash "$setup"
    as_root apt-get install -y nodejs
  fi
  command -v npm >/dev/null 2>&1 || as_root apt-get install -y npm
  mkdir -p "$NODE_HOME"
  npm install --prefix "$NODE_HOME" --no-audit --no-fund @earendil-works/pi-coding-agent@1.4.0
  [[ -f "$PI_CLI" ]] || die "Pi Agent CLI 未安装：$PI_CLI"
}

write_pi_config() {
  mkdir -p "$PI_AGENT_DIR" "$PI_WORKSPACE_DIR" "$PI_CIX_HOME/sessions" "$LOG_DIR" "$PID_DIR"
  cat > "$PI_AGENT_DIR/models.json" <<'JSON'
{"providers":{"mnn":{"name":"MNN local","baseUrl":"http://127.0.0.1:8000/v1","api":"openai-completions","apiKey":"mnn-local","models":[{"id":"Qwen3.5-4B-MNN","name":"Qwen3.5 4B MNN","baseUrl":"http://127.0.0.1:8000/v1","reasoning":false,"input":["text"],"contextWindow":32768,"maxTokens":8192}]}}}
JSON
  cat > "$PI_AGENT_DIR/settings.json" <<'JSON'
{"$schema":"https://pi.dev/schemas/settings.schema.json","defaultProvider":"mnn","defaultModel":"Qwen3.5-4B-MNN","defaultThinkingLevel":"off","lastChangelogVersion":"1.4.0"}
JSON
}

install_all() {
  mkdir -p "$PI_CIX_HOME"
  if [[ ! -f "$ENV_FILE" ]]; then
    cp "$SCRIPT_DIR/config.env.example" "$ENV_FILE"
  fi
  set -a
  . "$ENV_FILE"
  set +a
  # Recompute defaults after reading the editable deployment environment.
  PI_CIX_HOME="${PI_CIX_HOME:-$HOME/.local/share/pi-agent-cix}"
  MODEL_ROOT="${MODEL_ROOT:-$PI_CIX_HOME/models}"
  ASR_MODEL_DIR="${ASR_MODEL_DIR:-$MODEL_ROOT/onnx_sensevoice}"
  TTS_MODEL_DIR="${TTS_MODEL_DIR:-$MODEL_ROOT/onnx_kokoro}"
  QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_ROOT/Qwen3.5-4B-MNN}"
  MNN_SOURCE_DIR="${MNN_SOURCE_DIR:-$PI_CIX_HOME/source/MNN}"
  MNN_BUILD_DIR="${MNN_BUILD_DIR:-$MNN_SOURCE_DIR/build-cix-opencl}"
  MNNCLI_BUILD_DIR="${MNNCLI_BUILD_DIR:-$MNN_SOURCE_DIR/apps/mnncli/build_mnncli}"
  MNNCLI="$MNNCLI_BUILD_DIR/mnncli"
  OPENCL_RUNTIME_LIB="${OPENCL_RUNTIME_LIB:-/opt/cixgpu-pro/lib/aarch64-linux-gnu}"
  CIX_WHEEL_DIR="${CIX_WHEEL_DIR:-/usr/share/cix/pypi}"
  CIX_MODEL_REPO="${CIX_MODEL_REPO:-cix/ai_model_hub_25_Q3}"
  QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-MNN/Qwen3.5-4B-MNN}"
  MNN_API_HOST="${MNN_API_HOST:-127.0.0.1}"
  MNN_BACKEND_PORT="${MNN_BACKEND_PORT:-8001}"
  MNN_PROXY_PORT="${MNN_PROXY_PORT:-8000}"
  PI_PROVIDER="${PI_PROVIDER:-mnn}"
  PI_MODEL="${PI_MODEL:-Qwen3.5-4B-MNN}"
  PI_TOOLS="${PI_TOOLS:-read,grep,find,ls}"
  PI_WORKSPACE_DIR="${PI_WORKSPACE_DIR:-$PI_CIX_HOME/workspace}"
  PI_SESSION_FILE="${PI_SESSION_FILE:-$PI_CIX_HOME/sessions/voice.jsonl}"
  PI_AGENT_DIR="$PI_CIX_HOME/pi-agent"
  NODE_HOME="$PI_CIX_HOME/node"
  PI_CLI="$NODE_HOME/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
  PYTHON="$PI_CIX_HOME/venv/bin/python"
  LOG_DIR="$PI_CIX_HOME/logs"
  PID_DIR="$PI_CIX_HOME/run"
  check_board
  install_system_packages
  check_board
  install_python_packages
  download_models
  build_mnn
  install_node_and_pi
  write_pi_config
  mkdir -p "$PI_CIX_HOME/src"
  install -m 755 "$SCRIPT_DIR/src/voice_assistant.py" "$PI_CIX_HOME/src/voice_assistant.py"
  install -m 755 "$SCRIPT_DIR/src/voice_assistant_pi.py" "$PI_CIX_HOME/src/voice_assistant_pi.py"
  install -m 644 "$SCRIPT_DIR/src/pi_mnn_openai_proxy.js" "$PI_CIX_HOME/src/pi_mnn_openai_proxy.js"
  printf '\n安装完成。运行：\n  %s start\n  %s voice\n' "$SCRIPT_DIR/deploy.sh" "$SCRIPT_DIR/deploy.sh"
}

pid_alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

start_one() {
  local name="$1"; shift
  local pidfile="$PID_DIR/$name.pid" logfile="$LOG_DIR/$name.log"
  if pid_alive "$pidfile"; then printf '%s 已运行（PID %s）\n' "$name" "$(cat "$pidfile")"; return; fi
  rm -f "$pidfile"
  nohup "$@" >>"$logfile" 2>&1 </dev/null &
  printf '%s\n' "$!" > "$pidfile"
  printf '%s 已启动（PID %s），日志：%s\n' "$name" "$(cat "$pidfile")" "$logfile"
}

wait_backend() {
  local timeout="${BACKEND_START_TIMEOUT:-300}" elapsed=0
  while ! curl -fsS --max-time 2 "http://$MNN_API_HOST:$MNN_BACKEND_PORT/v1/models" >/dev/null; do
    if ! pid_alive "$PID_DIR/mnn-backend.pid"; then
      tail -n 60 "$LOG_DIR/mnn-backend.log" >&2 || true
      die "MNN 后端启动失败；请查看 $LOG_DIR/mnn-backend.log"
    fi
    (( elapsed < timeout )) || die "等待 MNN 模型启动超时；请查看 $LOG_DIR/mnn-backend.log"
    sleep 1
    elapsed=$((elapsed + 1))
  done
}

start_services() {
  [[ -x "$MNNCLI" ]] || die "找不到 mnncli，先运行 deploy.sh install"
  [[ -f "$QWEN_MODEL_DIR/config-opencl.json" ]] || die "找不到 Qwen3.5-4B 模型配置"
  mkdir -p "$LOG_DIR" "$PID_DIR"
  load_runtime_env
  start_one mnn-backend env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$MNNCLI" serve \
    --config "$QWEN_MODEL_DIR/config-opencl.json" --thinking off --host "$MNN_API_HOST" --port "$MNN_BACKEND_PORT"
  wait_backend
  start_one pi-proxy env MNN_BACKEND_URL="$MNN_BACKEND_URL" MNN_PROXY_HOST="$MNN_PROXY_HOST" \
    MNN_PROXY_PORT="$MNN_PROXY_PORT" node "$PI_CIX_HOME/src/pi_mnn_openai_proxy.js"
}

stop_one() {
  local pidfile="$PID_DIR/$1.pid"
  if pid_alive "$pidfile"; then kill "$(cat "$pidfile")" 2>/dev/null || true; rm -f "$pidfile"; printf '%s 已停止\n' "$1"
  else rm -f "$pidfile"; printf '%s 未运行\n' "$1"; fi
}

usage() {
  cat <<EOF
用法：$SCRIPT_DIR/deploy.sh [install|start|stop|restart|status|voice|logs]
  install   安装 Python、Pi Agent、模型和 MNN OpenCL 服务
  start     启动本机 MNN 服务及工具调用适配代理
  stop      停止服务
  restart   重启服务
  status    查看进程及本地 API
  voice     启动麦克风 → ASR → Pi Agent → TTS
  logs      跟踪后端和代理日志
EOF
}

mkdir -p "$LOG_DIR" "$PID_DIR"
case "${1:-install}" in
  install) install_all ;;
  start) start_services ;;
  stop) stop_one pi-proxy; stop_one mnn-backend ;;
  restart) stop_one pi-proxy; stop_one mnn-backend; start_services ;;
  status)
    for name in mnn-backend pi-proxy; do
      if pid_alive "$PID_DIR/$name.pid"; then printf '%s: running (PID %s)\n' "$name" "$(cat "$PID_DIR/$name.pid")"
      else printf '%s: stopped\n' "$name"; fi
    done
    curl -fsS --max-time 2 "http://$MNN_API_HOST:$MNN_PROXY_PORT/v1/models" || true
    printf '\n'
    ;;
  voice)
    start_services
    load_runtime_env
    [[ -f "$PI_CLI" ]] || die "找不到 Pi Agent CLI"
    shift
    exec "$PYTHON" "$PI_CIX_HOME/src/voice_assistant_pi.py" \
      --pi-node "$(command -v node)" --pi-cli "$PI_CLI" --provider "$PI_PROVIDER" \
      --model "$PI_MODEL" --tools "$PI_TOOLS" --session-file "$PI_SESSION_FILE" "$@"
    ;;
  logs) tail -n 100 -F "$LOG_DIR/mnn-backend.log" "$LOG_DIR/pi-proxy.log" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
