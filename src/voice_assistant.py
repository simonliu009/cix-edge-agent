#!/usr/bin/env python3
"""Always-on SenseVoice -> Pi local LLM -> Kokoro voice assistant."""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import requests

ASR_ROOT = Path(os.getenv("ASR_MODEL_DIR", "/root/ai_model_hub/models/Audio/Speech_Recognition/onnx_sensevoice"))
TTS_ROOT = Path(os.getenv("TTS_MODEL_DIR", "/root/ai_model_hub/models/Audio/TTS/onnx_kokoro"))
PI_DIR = Path(os.getenv("PI_CODING_AGENT_DIR", str(Path.home() / ".pi" / "agent")))
SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2
TTS_SAMPLE_RATE = 24000


def alsa_devices(command: str) -> list[tuple[int, int, str]]:
    """Return (card, device, descriptive line) entries from arecord/aplay -l."""
    if not shutil.which(command):
        return []
    try:
        result = subprocess.run([command, "-l"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices = []
    for line in result.stdout.splitlines():
        match = re.search(r"card\s+(\d+):.*?device\s+(\d+):", line, re.IGNORECASE)
        if match:
            devices.append((int(match.group(1)), int(match.group(2)), line.strip()))
    return devices


def choose_audio_device(command: str, override: str | None) -> str:
    if override:
        return override
    devices = alsa_devices(command)
    for card, device, label in devices:
        if "jabra" in label.lower() or "speak 710" in label.lower():
            return f"plughw:{card},{device}"
    # Prefer a discovered ALSA device, otherwise let ALSA use its configured default.
    if devices:
        card, device, _ = devices[0]
        return f"plughw:{card},{device}"
    return "default"


def read_pi_llm_config() -> dict[str, str]:
    settings_path = PI_DIR / "settings.json"
    models_path = PI_DIR / "models.json"
    settings: dict[str, Any] = {}
    models: dict[str, Any] = {}
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    try:
        models = json.loads(models_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    provider_name = settings.get("defaultProvider", "mnn")
    provider = models.get("providers", {}).get(provider_name, {})
    models_for_provider = provider.get("models", [])
    model_override = os.getenv("LLM_MODEL")
    model_id = model_override or settings.get("defaultModel", "")
    model_info = next((m for m in models_for_provider if m.get("id") == model_id), {})

    # Model selection follows Pi's default model; local MNN profiles share port 8000.
    return {
        "base_url": os.getenv("LLM_BASE_URL") or model_info.get("baseUrl") or provider.get("baseUrl") or "http://127.0.0.1:8000/v1",
        "model": model_override or model_id or "Qwen3.5-9B-MNN",
        "api_key": os.getenv("LLM_API_KEY") or provider.get("apiKey") or "",
        "api": provider.get("api", "openai-completions"),
    }


def start_recorder(device: str) -> subprocess.Popen:
    if not shutil.which("arecord"):
        raise RuntimeError("找不到 arecord。请安装 ALSA 工具包 alsa-utils。")
    command = [
        "arecord", "-q", "-D", device, "-t", "raw", "-f", "S16_LE",
        "-r", str(SAMPLE_RATE), "-c", "1", "--period-size", str(CHUNK_SAMPLES),
    ]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    except OSError as exc:
        raise RuntimeError(f"无法启动 arecord：{exc}") from exc
    # Give ALSA a moment to open the selected device and report immediate failures.
    time.sleep(0.15)
    if proc.poll() is not None:
        proc.stdout.close()
        raise RuntimeError(
            f"arecord 无法打开麦克风设备 {device!r}。检查 Jabra 是否连接，或用 --mic-device 指定 ALSA 设备。"
        )
    return proc


def stop_recorder(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    if proc.stdout:
        proc.stdout.close()


def listen_for_utterance(
    proc: subprocess.Popen,
    threshold: float,
    silence_seconds: float,
    max_seconds: float,
    min_seconds: float,
    noise_multiplier: float,
) -> np.ndarray | None:
    """Read continuously and use energy VAD to return one speech utterance."""
    if proc.stdout is None:
        raise RuntimeError("arecord 没有提供音频流")
    preroll: collections.deque[np.ndarray] = collections.deque(maxlen=15)  # 300 ms
    utterance: list[np.ndarray] = []
    speech_started = False
    speech_run = 0
    silence_run = 0
    elapsed_chunks = 0
    silence_chunks = max(1, int(silence_seconds * 1000 / CHUNK_MS))
    max_chunks = max(1, int(max_seconds * 1000 / CHUNK_MS))
    min_chunks = max(1, int(min_seconds * 1000 / CHUNK_MS))
    noise_rms = 0.002
    pending = bytearray()

    while True:
        while len(pending) < CHUNK_BYTES:
            part = proc.stdout.read(CHUNK_BYTES - len(pending))
            if not part:
                code = proc.poll()
                raise RuntimeError(f"麦克风音频流结束（arecord 返回码 {code}），设备可能已断开。")
            pending.extend(part)
        raw = bytes(pending[:CHUNK_BYTES])
        del pending[:CHUNK_BYTES]
        samples_i16 = np.frombuffer(raw, dtype="<i2").copy()
        frame = samples_i16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(frame * frame)))
        elapsed_chunks += 1
        dynamic_threshold = max(threshold, noise_rms * noise_multiplier)
        active = rms >= dynamic_threshold

        if not speech_started:
            preroll.append(frame)
            if active:
                speech_run += 1
            else:
                speech_run = 0
                # Adapt to ambient sound only while the VAD is idle.
                noise_rms = 0.96 * noise_rms + 0.04 * rms
            if speech_run >= 3:  # 60 ms above threshold
                speech_started = True
                utterance.extend(list(preroll))
                silence_run = 0
                print("\n检测到语音，正在聆听…", flush=True)
        else:
            utterance.append(frame)
            if active:
                silence_run = 0
            else:
                silence_run += 1
            if silence_run >= silence_chunks or len(utterance) >= max_chunks:
                # Keep a short natural tail, but remove most of the silence used to close VAD.
                keep_chunks = min(silence_run, max(1, int(0.18 * 1000 / CHUNK_MS)))
                if silence_run > keep_chunks:
                    utterance = utterance[:-silence_run + keep_chunks]
                if len(utterance) < min_chunks:
                    return None
                return np.concatenate(utterance).astype(np.float32, copy=False)


def make_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    parts = [
        "你是一个通过语音交互的中文助手。用自然、简洁的口语回答，通常控制在一到三句话。不要输出 Markdown、代码块或语音无法朗读的格式。",
    ]
    for user, assistant in history[-4:]:
        parts.append(f"\n用户：{user}\n助手：{assistant}")
    parts.append(f"\n用户：{user_text}\n助手：")
    return "".join(parts)


def response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"大模型 endpoint 返回了空 choices：{payload}")
    choice = choices[0]
    text = choice.get("text")
    if text is None:
        text = (choice.get("message") or {}).get("content", "")
    if isinstance(text, list):
        text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in text)
    text = str(text).strip()
    text = re.sub(r"^\s*(助手|assistant)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return text


def ask_llm(
    config: dict[str, str],
    user_text: str,
    history: list[tuple[str, str]],
    timeout: float,
    max_tokens: int,
    temperature: float,
) -> str:
    base = config["base_url"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if config.get("api_key"):
        headers["Authorization"] = f"Bearer {config['api_key']}"
    prompt = make_prompt(history, user_text)
    api_style = config.get("api", "openai-completions")
    completion_url = base if base.endswith("/completions") else f"{base}/completions"
    chat_url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    if api_style in ("openai-completions", "openai-completion"):
        body = {
            "model": config["model"], "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False,
        }
        result = requests.post(completion_url, headers=headers, json=body, timeout=timeout)
        if result.status_code not in (404, 405, 422):
            result.raise_for_status()
            return response_text(result.json())

    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "你是一个通过语音交互的中文助手。用自然、简洁的口语回答，通常控制在一到三句话。不要输出 Markdown、代码块或语音无法朗读的格式。"},
            *[{"role": role, "content": content} for user, assistant in history[-4:] for role, content in (("user", user), ("assistant", assistant))],
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }
    result = requests.post(chat_url, headers=headers, json=body, timeout=timeout)
    result.raise_for_status()
    return response_text(result.json())


def generate_audio(text: str, pipelines: dict[str, Any], voices: dict[str, Path]) -> Iterator[np.ndarray]:
    """Yield native Kokoro generator chunks without joining sentence waveforms."""
    use_chinese = any("\u3400" <= ch <= "\u9fff" for ch in text)
    lang = "zh" if use_chinese else "en"
    voice_path = str(voices[lang])
    yielded = False
    for _graphemes, _phonemes, audio in pipelines[lang](text, voice=voice_path):
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            yielded = True
            yield chunk
    if not yielded:
        raise RuntimeError("Kokoro 没有生成音频")


def play_audio(waveforms: Iterable[np.ndarray], device: str) -> None:
    """Open ALSA once, then feed each generated sentence directly to playback."""
    if not shutil.which("aplay"):
        raise RuntimeError("找不到 aplay。请安装 ALSA 工具包 alsa-utils。")
    command = [
        "aplay", "-q", "-D", device, "-t", "raw", "-f", "S16_LE",
        "-r", str(TTS_SAMPLE_RATE), "-c", "1",
    ]
    player = None
    try:
        player = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)
        assert player.stdin is not None
        chunk_count = 0
        for waveform in waveforms:
            if player.poll() is not None:
                detail = player.stderr.read().decode(errors="replace").strip() if player.stderr else ""
                raise RuntimeError(f"无法通过 {device!r} 播放音频：{detail or player.returncode}")
            pcm = np.rint(np.asarray(waveform, dtype=np.float32) * 32768.0).astype("<i2").tobytes()
            if pcm:
                player.stdin.write(pcm)
                player.stdin.flush()
                chunk_count += 1
        if chunk_count == 0:
            raise RuntimeError("Kokoro 没有生成可播放的音频")
        player.stdin.close()
        try:
            code = player.wait(timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("播放超时，请检查扬声器设备") from exc
        detail = player.stderr.read().decode(errors="replace").strip() if player.stderr else ""
        if code != 0:
            raise RuntimeError(f"无法通过 {device!r} 播放音频：{detail or code}")
    except OSError as exc:
        raise RuntimeError(f"无法启动扬声器播放：{exc}") from exc
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
            try:
                player.wait(timeout=2)
            except subprocess.TimeoutExpired:
                player.kill()
                player.wait(timeout=2)
        if player is not None and player.stdin is not None and not player.stdin.closed:
            player.stdin.close()


def has_spoken_content(text: str) -> bool:
    # Discard ASR results containing only punctuation, whitespace, or symbols.
    return any(char.isalnum() for char in text)


def load_models() -> tuple[Any, Any, dict[str, Any], dict[str, Path]]:
    for path in (ASR_ROOT, TTS_ROOT):
        if not path.is_dir():
            raise RuntimeError(f"找不到模型目录：{path}")
        sys.path.insert(0, str(path))

    from funasr_onnx import SenseVoiceSmall
    from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
    from kokoro import KPipeline, KModel

    previous = Path.cwd()
    try:
        os.chdir(ASR_ROOT)
        asr = SenseVoiceSmall(str(ASR_ROOT / "sense_voice_mask256.cix"), infer_config={"backend": "NPU", "max_len": 256})
        os.chdir(TTS_ROOT)
        tts_model = KModel(str(TTS_ROOT / "datasets/npu_config.json"), use_npu=True)
        pipelines = {
            "en": KPipeline(lang_code="a", model=tts_model),
            "zh": KPipeline(lang_code="z", model=tts_model),
        }
    except Exception:
        try:
            if "asr" in locals():
                asr.clean()
        finally:
            if "tts_model" in locals():
                tts_model.clean()
        raise
    finally:
        os.chdir(previous)

    voices = {"en": TTS_ROOT / "voices/af_heart.pt", "zh": TTS_ROOT / "voices/zf_xiaoyi.pt"}
    for voice in voices.values():
        if not voice.is_file():
            raise RuntimeError(f"找不到 Kokoro 音色文件：{voice}")
    return asr, rich_transcription_postprocess, pipelines, voices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="常驻语音助手：SenseVoice ASR -> Pi 大模型 -> Kokoro TTS")
    parser.add_argument("--mic-device", help="ALSA 麦克风设备，例如 plughw:1,0；默认自动寻找 Jabra SPEAK 710")
    parser.add_argument("--speaker-device", help="ALSA 扬声器设备，例如 plughw:1,0；默认自动寻找 Jabra SPEAK 710")
    parser.add_argument("--endpoint", default=None, help="OpenAI 兼容 endpoint base URL，默认读取 Pi 配置")
    parser.add_argument("--model", default=None, help="大模型 ID，默认读取 Pi 配置")
    parser.add_argument("--api-key", default=None, help="endpoint API key，默认读取 Pi 配置")
    parser.add_argument("--silence-seconds", type=float, default=0.8, help="连续静音多久结束一句，默认 0.8 秒")
    parser.add_argument("--max-utterance-seconds", type=float, default=20.0, help="单句最长录音时长")
    parser.add_argument("--min-utterance-seconds", type=float, default=0.2, help="短于此长度的检测片段忽略")
    parser.add_argument("--vad-threshold", type=float, default=0.008, help="VAD 最低 RMS 阈值，范围 0 到 1")
    parser.add_argument("--vad-noise-multiplier", type=float, default=3.0, help="相对环境噪声的 VAD 倍数")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="大模型 HTTP 超时秒数")
    parser.add_argument("--max-tokens", type=int, default=256, help="大模型回答 token 上限")
    parser.add_argument("--temperature", type=float, default=0.6, help="大模型采样温度")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_pi_llm_config()
    if args.endpoint:
        config["base_url"] = args.endpoint
    if args.model:
        config["model"] = args.model
    if args.api_key:
        config["api_key"] = args.api_key

    mic_device = choose_audio_device("arecord", args.mic_device)
    speaker_device = choose_audio_device("aplay", args.speaker_device)
    print(f"麦克风：{mic_device}；扬声器：{speaker_device}")
    print(f"大模型：{config['model']} @ {config['base_url']}")
    print("正在加载 SenseVoice 与 Kokoro NPU 模型…")

    asr = None
    pipelines = None
    try:
        asr, postprocess, pipelines, voices = load_models()
        print("模型已就绪。VAD 正在持续监听；说完停顿片刻即可，按 Ctrl+C 退出。")
        history: list[tuple[str, str]] = []
        while True:
            recorder = None
            try:
                recorder = start_recorder(mic_device)
                utterance = listen_for_utterance(
                    recorder, args.vad_threshold, args.silence_seconds,
                    args.max_utterance_seconds, args.min_utterance_seconds,
                    args.vad_noise_multiplier,
                )
            finally:
                stop_recorder(recorder)

            if utterance is None:
                continue
            raw_result = asr(utterance, language="auto", textnorm="withitn")
            recognized = postprocess(raw_result[0] if isinstance(raw_result, (list, tuple)) else raw_result).strip()
            if not has_spoken_content(recognized):
                print("忽略空白或仅含标点的识别结果，继续监听。", flush=True)
                continue
            print(f"\n你：{recognized}", flush=True)

            try:
                answer = ask_llm(config, recognized, history, args.request_timeout, args.max_tokens, args.temperature)
            except requests.RequestException as exc:
                print(f"大模型请求失败：{exc}。继续监听。", file=sys.stderr, flush=True)
                continue
            if not answer:
                print("大模型返回空文本，继续监听。", flush=True)
                continue
            print(f"助手：{answer}", flush=True)
            history.append((recognized, answer))
            history = history[-6:]

            try:
                play_audio(generate_audio(answer, pipelines, voices), speaker_device)
                time.sleep(0.25)  # Let the Jabra speaker settle before reopening its microphone.
            except Exception as exc:
                print(f"TTS/播放失败：{exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\n已停止监听。")
        return 0
    except Exception as exc:
        print(f"启动或运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
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
