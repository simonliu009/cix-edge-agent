# Pi Agent + CIX ASR/TTS + Qwen3.5-4B 宿主机部署包

此目录提供 Orange Pi 6 Plus / CIX 平台上的本地语音 Agent 部署材料。语音识别和语音合成使用 CIX NPU；Qwen3.5-4B 使用 MNN OpenCL / Mali GPU。

**核心技术创新是自行实现/适配 MNN 到 OpenAI Compatible HTTP endpoint 的服务层。** 我们在 MNN CLI 服务端提供本地模型列表和聊天补全接口，支持普通 JSON 回复与 SSE 流式回复，并处理模型 ID、文本内容数组和逐请求 thinking 开关。MNN HTTP 服务监听 127.0.0.1:8001；Pi Agent 通过监听 127.0.0.1:8000 的 Node.js 代理访问它。Node.js 层专门转换 Pi Agent 与 Qwen 的工具调用格式，两层分工明确。


## 目录内容

- deploy.sh：宿主机依赖、模型下载、MNN 编译、服务启动和语音入口
- requirements.txt：语音、推理和模型下载 Python 依赖
- config.env.example：模型仓库、设备路径和端口配置模板
- src/voice_assistant_pi.py：SenseVoice → Pi Agent RPC → Kokoro 主循环
- src/voice_assistant.py：音频设备、VAD、CIX ASR/TTS 加载与播放
- src/pi_mnn_openai_proxy.js：Pi Agent 工具调用与 MNN API 格式适配
- patches/mnncli-cix-openai.patch：本项目对 MNN CLI HTTP 服务的自研适配代码，包含 OpenAI 兼容请求与响应行为调整

## 核心实现：MNN → OpenAI Compatible HTTP Endpoint

这是项目的核心技术创新和聊天链路的模型服务边界：Pi Agent 不直接链接 MNN；它通过 Node.js 适配代理访问本机 OpenAI 兼容 HTTP 接口，接口再调用 MNN 模型。

- 模型发现：GET /v1/models 返回服务实际加载的模型 ID。
- 对话请求：POST /v1/chat/completions 接收 role/messages、model 和 stream 等字段。
- 返回模式：提供普通 JSON 完成结果与 OpenAI 风格 SSE 流式输出。
- 请求兼容：接受字符串消息内容，也处理文本内容数组；按请求布尔字段控制 Qwen thinking。
- 并发保护：推理期间在 MNN LLM 访问锁内更新 thinking 配置并执行生成，避免请求间配置交叉。
- 本地边界：默认监听 127.0.0.1:8001；Pi Agent 使用的 127.0.0.1:8000 由 Node 代理提供。

相关实现位于 MNN CLI 的 mnncli_server.hpp、mnncli_server.cpp 和 serve_command_handler.cpp；deploy.sh 会检出固定 MNN 源码版本，并应用 patches/mnncli-cix-openai.patch。Node 代理的工具调用转换在 src/pi_mnn_openai_proxy.js 中，负责将 Pi Agent 的结构化工具定义转换成 Qwen 工具提示格式，并把模型输出解析回工具调用。

## 运行环境

部署脚本按以下环境设计：

- Orange Pi 6 Plus 或兼容的 CIX AArch64 主机
- CIX 官方 Linux 系统，已提供 aipu 内核驱动、/dev/aipu* 和 CIX NPU 用户态运行库
- CIX OpenCL/Mali 用户态库，默认位于 /opt/cixgpu-pro/lib/aarch64-linux-gnu
- 官方 Python wheel 目录，默认位于 /usr/share/cix/pypi
- 可用的麦克风和扬声器；ALSA 能发现设备
- 网络连接，用于下载 MNN、Pi Agent 和选定的模型资产
- 足够的磁盘空间存放模型、构建依赖和 MNN 编译产物。Qwen3.5-4B 权重本身约 2.6 GB

部署脚本不会替换或编译板卡内核驱动。缺少 NPU 设备节点、NOE 运行库或 Mali OpenCL 库时，脚本会停止并提示先准备 CIX 官方系统环境。

## 一键安装

在目标板上取得本目录后运行：

    cd pi-agent-cix-host
    chmod +x deploy.sh
    ./deploy.sh install

安装步骤会：

1. 检查 AArch64、/dev/aipu*、/dev/mali0 和 CIX operator 目录。
2. 安装编译、音频、OpenCL 等系统依赖。
3. 创建 Python 3.11 环境。系统没有 Python 3.11 时，使用 Miniforge 创建环境。
4. 安装 Python 依赖及板卡自带的 libnoe、NOE_Engine、onnxruntime_zhouyi、ZhouyiOperators wheel。
5. 通过 ModelScope 的文件筛选下载 SenseVoice、Kokoro 需要的运行文件，不下载 CIX AI Model Hub 全仓库。
6. 下载 MNN/Qwen3.5-4B-MNN 中的量化权重、tokenizer 和 OpenCL 配置。
7. 获取固定版本的 MNN 源码，应用本目录补丁，编译 MNN OpenCL 和 mnncli。
8. 安装 Node.js 22.19 或更新版本及 Pi Agent 1.4.0，并写入独立的 Pi provider 配置。

首次安装包含模型下载和 C++ 编译，耗时取决于网络与开发板存储。安装命令需要 sudo 权限以安装系统包。

## 启动与操作

启动自研 MNN OpenAI 兼容 HTTP endpoint 和 Pi Agent 工具适配代理：

    ./deploy.sh start

开始麦克风语音交互：

    ./deploy.sh voice

查看状态、日志和停止服务：

    ./deploy.sh status
    ./deploy.sh logs
    ./deploy.sh stop

选择音频设备时，可把 ALSA 设备参数传给语音入口：

    ./deploy.sh voice --mic-device plughw:1,0 --speaker-device plughw:1,0

运行配置在用户目录：

    ~/.local/share/pi-agent-cix/config.env

首次安装后可编辑该文件修改模型仓库、目录或本机端口，然后再次运行 install 或 restart。默认 API 地址为 http://127.0.0.1:8000/v1；MNN 原始后端使用 127.0.0.1:8001。两个服务均只绑定本机地址。

## Agent 工具与会话

Pi Agent 默认只允许 read、grep、find、ls 四种只读文件工具，并关闭 MCP、扩展、技能和提示模板。默认工作目录为部署目录下的 workspace。会话写入本机 sessions/voice.jsonl，跨轮保留上下文。

会话中可能包含用户转写文本和模型回复。共享设备前，请先检查并按需删除会话文件。需要开放其他工具时，修改 config.env 中的 PI_TOOLS 并重新启动语音入口；增加写入或命令工具前，应先确认工作目录和系统权限。

## 模型与运行时来源

- CIX ASR/TTS 模型仓库默认使用 cix/ai_model_hub_25_Q3，并仅下载本项目所需文件。可在 config.env 设置 CIX_MODEL_REPO。
- Qwen3.5-4B-MNN 使用 ModelScope 仓库 MNN/Qwen3.5-4B-MNN。
- CIX NPU Python wheel 直接使用目标系统已安装的 /usr/share/cix/pypi 目录。
- MNN 源码固定在 dbf9ba61b08106a3b43426e295d1a29cb41cc8fe，以便补丁可复现。
- Pi Agent 固定安装 @earendil-works/pi-coding-agent 1.4.0。

## 许可证与发布提示

部署脚本从上游模型仓库下载权重，不把权重复制进本目录。SenseVoice、Kokoro、Qwen3.5 MNN 权重、CIX 运行库、MNN、Pi Agent 和 Python 依赖分别遵循其上游许可证与分发条件。公开代码仓库前，应为本项目代码选定 LICENSE，并检查所有脚本、补丁、模型资产及第三方依赖的许可证；不要把上游模型权重误标成本项目代码许可证。

## 常见问题

**安装提示找不到 /dev/aipu***  
确认板卡运行的是带 CIX NPU 驱动的官方系统，重新启动后检查设备节点。该脚本不负责安装内核模块或刷写固件。

**找不到 ONNX Runtime operator 或 NOE wheel**  
检查 CIX 官方运行库是否安装，以及 /usr/share/cix/lib/onnxruntime/operator 和 /usr/share/cix/pypi 是否存在。Python 环境需要 3.11，以匹配板卡提供的 cp311 wheel。

**ASR/TTS 运行时找不到共享库**  
deploy.sh 会设置 LD_LIBRARY_PATH 和 OPERATOR_PATH。若手动运行 Python 程序，也需要加载部署脚本所设置的环境变量。

**麦克风打不开或录音设备选择错误**  
先用 arecord -l 和 aplay -l 查看 ALSA 设备，再将对应 plughw:CARD,DEVICE 参数传给 voice 命令。

**模型下载失败**  
检查 ModelScope 网络访问和仓库权限。可在 config.env 里替换模型仓库镜像；下载筛选路径按对应仓库目录结构匹配。

**Agent 无法连接模型**  
查看 logs/mnn-backend.log 与 logs/pi-proxy.log，并确认本机 8000、8001 端口未被占用。Pi Agent 模型配置由安装脚本写入独立目录，不会修改用户原有的 ~/.pi/agent 配置。

## 参考资料

- 项目提供的模型部署参考：[2026 Agentic AI 文档](https://2026agentic-ai.readthedocs.io/en/latest/)
- Orange Pi 6 Plus CIX Linux 系统用户手册：[官方手册 PDF](https://orangepi.net/wp-content/uploads/2026/01/OrangePi_6_Plus_Cix_Linux-System_User-Manual_v0.7.pdf)
- ModelScope Qwen3.5-4B-MNN：[模型仓库](https://www.modelscope.cn/models/MNN/Qwen3.5-4B-MNN)
- CIX AI Model Hub：[模型仓库](https://www.modelscope.cn/models/cix/ai_model_hub_25_Q3)
- Pi Agent：[上游代码仓库](https://github.com/earendil-works/pi)
- MNN：[上游代码仓库](https://github.com/alibaba/MNN)

