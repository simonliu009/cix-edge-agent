#!/usr/bin/env python3
"""Always-on voice assistant using Pi RPC: SenseVoice -> Pi Agent -> Kokoro."""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import voice_assistant as audio

NODE_BIN = Path(os.getenv("PI_NODE") or shutil.which("node") or "node")
PI_CLI = Path(os.getenv("PI_CLI", str(Path.home() / ".local/share/pi-agent-cix/node/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js")))
SPOKEN_SYSTEM_PROMPT = (
    "The user is speaking through a microphone. Reply naturally and concisely in Chinese by default, "
    "or in the language the user used. Keep the final answer easy to listen to. Avoid Markdown tables, "
    "code fences, and visual formatting unless the user explicitly asks for them."
)


class PiAgent:
    """Persistent Pi RPC subprocess. The agent owns model calls and conversation context."""

    def __init__(self, node: Path, cli: Path, provider: str, model: str, cwd: Path, timeout: float,
                 tools: str, session_file: Path, agent_dir: Path):
        node_path = Path(shutil.which(str(node)) or node)
        if not node_path.is_file():
            raise RuntimeError(f"找不到 Pi 所需的 Node.js：{node}")
        if not cli.is_file():
            raise RuntimeError(f"找不到 Pi Agent CLI：{cli}")
        self.timeout = timeout
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=80)
        child_env = os.environ.copy()
        child_env["PATH"] = f"{node_path.parent}{os.pathsep}{child_env.get('PATH', '')}"
        child_env["PI_CODING_AGENT_DIR"] = str(agent_dir)
        child_env["PI_CODING_AGENT_SESSION_DIR"] = str(session_file.parent)
        command = [
            str(node_path), str(cli), "--mode", "rpc", "--session", str(session_file),
            "--session-dir", str(session_file.parent), "--tools", tools,
            "--no-mcp", "--no-extensions", "--no-skills", "--no-prompt-templates",
            "--no-context-files",
            "--provider", provider, "--model", model, "--thinking", "off",
            "--append-system-prompt", SPOKEN_SYSTEM_PROMPT,
        ]
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=child_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"无法启动 Pi Agent：{exc}") from exc
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()
        try:
            state = self._command_response({"type": "get_state"}, timeout=30)
            current_model = state.get("model") or {}
            print(f"Pi Agent 已连接：{current_model.get('provider', provider)}/{current_model.get('id', model)}")
        except Exception:
            self.close()
            raise

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in iter(self.process.stdout.readline, b""):
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(record, dict):
                    self.events.put(record)
        finally:
            self.events.put(None)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, b""):
            self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    def _send(self, command: dict[str, Any]) -> str:
        if self.process.poll() is not None:
            raise RuntimeError(self._process_error("Pi Agent 已退出"))
        request_id = str(uuid.uuid4())
        command["id"] = request_id
        if self.process.stdin is None:
            raise RuntimeError("Pi Agent stdin 不可用")
        try:
            self.process.stdin.write(json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(self._process_error(f"无法向 Pi Agent 发送命令：{exc}")) from exc
        return request_id

    def _next_record(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(self._process_error("等待 Pi Agent 响应超时"))
        try:
            record = self.events.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(self._process_error("等待 Pi Agent 响应超时")) from exc
        if record is None:
            raise RuntimeError(self._process_error("Pi Agent 输出流已关闭"))
        return record

    def _process_error(self, message: str) -> str:
        details = "\n".join(self.stderr_tail)
        return f"{message}" + (f"\nPi stderr：\n{details}" if details else "")

    def _command_response(self, command: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        request_id = self._send(command)
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            record = self._next_record(deadline)
            if record.get("type") == "response" and record.get("id") == request_id:
                if not record.get("success"):
                    raise RuntimeError(record.get("error") or "Pi Agent RPC 命令失败")
                return record.get("data") or {}

    def ask(self, recognized_text: str) -> str:
        prompt = f"用户通过麦克风说：\n{recognized_text}\n\n请理解并回应用户。"
        request_id = self._send({"type": "prompt", "message": prompt})
        deadline = time.monotonic() + self.timeout
        prompt_accepted = False
        settled = False
        while not settled:
            record = self._next_record(deadline)
            if record.get("type") == "response" and record.get("id") == request_id:
                if not record.get("success"):
                    raise RuntimeError(record.get("error") or "Pi Agent 拒绝了语音输入")
                prompt_accepted = True
            elif record.get("type") == "agent_settled":
                if record.get("aborted"):
                    raise RuntimeError("Pi Agent 已中止本轮回答")
                settled = True
        if not prompt_accepted:
            raise RuntimeError("Pi Agent 未确认收到语音输入")
        data = self._command_response({"type": "get_last_assistant_text"})
        answer = data.get("text")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Pi Agent 没有返回可朗读的文本")
        return answer.strip()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        for stream_name in ("stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass


def parse_args() -> argparse.Namespace:
    config = audio.read_pi_llm_config()
    parser = argparse.ArgumentParser(description="常驻语音助手：SenseVoice ASR -> Pi Agent -> Kokoro TTS")
    parser.add_argument("--mic-device", help="ALSA 麦克风设备，例如 plughw:1,0；默认自动寻找 Jabra SPEAK 710")
    parser.add_argument("--speaker-device", help="ALSA 扬声器设备，例如 plughw:1,0；默认自动寻找 Jabra SPEAK 710")
    parser.add_argument("--pi-node", type=Path, default=NODE_BIN, help="Pi 使用的 Node.js 可执行文件")
    parser.add_argument("--pi-cli", type=Path, default=PI_CLI, help="Pi Agent CLI 的 cli.js 路径")
    parser.add_argument("--provider", default=os.getenv("PI_PROVIDER", "mnn"), help="Pi provider，默认 mnn")
    parser.add_argument("--model", default=os.getenv("PI_MODEL") or config["model"], help="Pi 模型 ID，默认使用 Pi 配置中的默认模型（endpoint 为 8000）")
    parser.add_argument("--tools", default=os.getenv("PI_TOOLS", "read,grep,find,ls"), help="Pi Agent 工具白名单，默认只读文件工具")
    parser.add_argument("--session-file", type=Path, default=Path(os.getenv("PI_SESSION_FILE", str(Path.home() / ".local/share/pi-agent-cix/sessions/voice.jsonl"))), help="持久化 Pi 会话文件")
    parser.add_argument("--pi-timeout", type=float, default=180, help="等待 Pi Agent 完成一轮的超时秒数")
    parser.add_argument("--silence-seconds", type=float, default=0.8, help="连续静音多久结束一句")
    parser.add_argument("--max-utterance-seconds", type=float, default=20.0, help="单句最长录音时长")
    parser.add_argument("--min-utterance-seconds", type=float, default=0.2, help="短于此长度的检测片段忽略")
    parser.add_argument("--vad-threshold", type=float, default=0.008, help="VAD 最低 RMS 阈值，范围 0 到 1")
    parser.add_argument("--vad-noise-multiplier", type=float, default=3.0, help="相对环境噪声的 VAD 倍数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mic_device = audio.choose_audio_device("arecord", args.mic_device)
    speaker_device = audio.choose_audio_device("aplay", args.speaker_device)
    print(f"麦克风：{mic_device}；扬声器：{speaker_device}")
    print(f"Pi Agent 模型：{args.provider}/{args.model}")
    print("正在加载 SenseVoice 与 Kokoro NPU 模型…")

    asr = None
    pipelines = None
    pi_agent = None
    try:
        asr, postprocess, pipelines, voices = audio.load_models()
        workspace = Path(os.getenv("PI_WORKSPACE_DIR", str(Path.home() / ".local/share/pi-agent-cix/workspace"))).expanduser()
        agent_dir = Path(os.getenv("PI_CODING_AGENT_DIR", str(Path.home() / ".local/share/pi-agent-cix/pi-agent"))).expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        args.session_file.parent.mkdir(parents=True, exist_ok=True)
        pi_agent = PiAgent(args.pi_node, args.pi_cli, args.provider, args.model, workspace,
                           args.pi_timeout, args.tools, args.session_file.expanduser(), agent_dir)
        print("模型已就绪。VAD 正在持续监听；说完停顿片刻即可，按 Ctrl+C 退出。")
        while True:
            recorder = None
            try:
                recorder = audio.start_recorder(mic_device)
                utterance = audio.listen_for_utterance(
                    recorder, args.vad_threshold, args.silence_seconds,
                    args.max_utterance_seconds, args.min_utterance_seconds,
                    args.vad_noise_multiplier,
                )
            finally:
                audio.stop_recorder(recorder)

            if utterance is None:
                continue
            raw_result = asr(utterance, language="auto", textnorm="withitn")
            recognized = postprocess(raw_result[0] if isinstance(raw_result, (list, tuple)) else raw_result).strip()
            if not audio.has_spoken_content(recognized):
                print("忽略空白或仅含标点的识别结果，继续监听。", flush=True)
                continue
            print(f"\n你：{recognized}", flush=True)
            try:
                print("Pi Agent 正在处理…", flush=True)
                answer = pi_agent.ask(recognized)
            except Exception as exc:
                print(f"Pi Agent 调用失败：{exc}。继续监听。", file=sys.stderr, flush=True)
                continue
            print(f"Pi Agent：{answer}", flush=True)
            try:
                audio.play_audio(audio.generate_audio(answer, pipelines, voices), speaker_device)
                time.sleep(0.25)
            except Exception as exc:
                print(f"TTS/播放失败：{exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\n已停止监听。")
        return 0
    except Exception as exc:
        print(f"启动或运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if pi_agent is not None:
            pi_agent.close()
        if asr is not None:
            try:
                asr.clean()
            except Exception:
                pass
        if pipelines:
            model = next(iter(pipelines.values())).model
            if model is not None:
                try:
                    model.clean()
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
