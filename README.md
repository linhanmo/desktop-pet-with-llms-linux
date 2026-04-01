# XiaoMo 桌面宠物（离线本地 LLM + 语音合成）

一款基于 Qt + Live2D 的桌面宠物应用，支持本地离线 LLM 对话与离线 TTS（文字转语音），提供托盘菜单与全局快捷键，解压即用或从源码构建。

## 功能概览
- Live2D 看板娘：支持呼吸、眨眼、视线跟随、物理模拟（头发/衣物摆动），可设置屏幕显示、透明背景、全局置顶等。
- 本地离线 LLM 对话：内置 llama.cpp CLI 调用，支持 1.5B/7B 规模，风格可选 Original/Universal/Anime，System Prompt 可自定义。
- 离线 TTS 播报：集成 sherpa-onnx 生态的多种离线语音模型，支持多说话人（sid）、音量调节。
- 对话与气泡：角色旁显示对话气泡，支持不同气泡主题；聊天窗口支持清空历史、上下文条数与最大生成长度可调。
- 本地提醒：内置定时任务（每天固定时间或按间隔）弹出提示并可驱动动作组。
- 多语言与主题：支持简体中文、英文与主题切换；配置与对话历史持久化存储。

## 环境要求（运行）
- Windows 10/11（x64）
- 建议安装 7-Zip（用于解压 build-msvc 分卷）

## 环境要求（从源码构建）
- Git、CMake、Visual Studio 2022 Build Tools（含 VC 工具链）
- Miniconda（用于获取 Qt6）：建议创建 qt6env 并安装 qt6-main、ninja
- Windows PowerShell + winget（可选）：用于安装缺失依赖

## 下载与运行（解压即用）
本项目采用 **build-msvc.zip 分卷**发布：下载后解压即可运行。

1) 下载
- 下载 Release 中的 `build-msvc.zip.001` ~ `build-msvc.zip.012`（必须全部下载到同一目录）

2) 解压
- 安装 7-Zip（推荐）后，在该目录执行：

```cmd
7z x build-msvc.zip.001
```

或在资源管理器里右键 `build-msvc.zip.001` → 7-Zip → 解压。

3) 运行
- 解压完成后，进入 `release/` 目录，运行 `XiaoMo.exe`

## 资源与目录约定
应用运行时查找资源的根目录为 `release/` 目录下的 `res/`。

`cubism.zip` 解压后的 **Cubism SDK 放在 `sdk/` 下**（用于从源码构建或资源更新场景），不放在 `res/` 下。

目录示例：

```
release/
  XiaoMo.exe
  res/
    bin/            # 包含 llama-cli.exe/llama.exe 等运行器（用于本地 LLM）
    llm/            # 本地 LLM 模型（*.gguf）
    models/         # Live2D 模型集合（每个子目录一个模型）
    voice_deps/     # 离线语音模型与依赖（sherpa-onnx 等）
    i18n/, icons/   # 语言与图标资源
sdk/
  cubism/         # Cubism SDK（由 cubism.zip 解压得到）
```

Live2D 模型（`res/models/`）：
- 每个模型放在独立文件夹，如 `res/models/<模型名>/`
- 模型文件优先匹配 `*.model3.json`，若无则尝试 `*.model.json`，否则回退 `model3.json/model.json/index.json`
- 默认模型根目录（设置中可改）：`文档\\XiaoMo\\Models`，首次运行会自动使用其中的第一个模型

本地 LLM 模型（`res/llm/`）：
- 将 gguf 文件放入 `res/llm/1.5B/` 或 `res/llm/7B/`（可不分文件夹，程序会自动匹配）
- 风格匹配规则（文件名包含以下关键字之一将优先选择）：
  - Anime：包含 `anime` 或 `anime.q`
  - Original：包含 `original` 或 `llama`
  - Universal：包含 `universal`
- 支持环境变量覆盖：
  - `LLAMA_RUNNER` 指定 llama 运行器路径
  - `LLM_MODEL` 指定 gguf 模型文件路径

离线 TTS（`res/voice_deps/`）：
- 语音模型目录：`res/voice_deps/models/<模型名>/`，程序会自动列出可用模型
- 多说话人（sid）：若模型包含说话人映射（speaker_id_map 或 speakers.txt），可在“AI 设置”里调节 sid
- sherpa-onnx 可执行目录自动检测于 `res/voice_deps/sherpa-onnx-*/bin`

## 快捷键与托盘
- Ctrl+T：显示/隐藏聊天窗口
- Ctrl+H：显示/隐藏桌宠窗口
- Ctrl+S：打开设置窗口
- 托盘图标：右键菜单可快速打开聊天、设置、退出；切换显示/隐藏状态

## 使用指南（设置窗口）
设置窗口分“基础设置 / 模型设置 / AI 设置 / 高级设置”四个页签：

1) 基础设置
- 主题与语言：切换应用主题与显示语言
- 窗口行为：全局置顶、透明背景、鼠标穿透（穿透开启后需 Alt+Tab 切回再关闭）
- 输出气泡：选择角色旁边对话气泡样式（如 爱心/漫画）

2) 模型设置（Live2D）
- 模型选择：从默认模型根目录中选择不同模型；可一键打开当前模型所在目录
- 去除水印：为当前模型选择一个 `*.exp3.json` 表达式文件作为“水印表达式”，以便覆盖并去除水印效果（可随时取消）
- 动作与效果：开关“自动呼吸/自动眨眼/视线跟踪/物理模拟”

3) AI 设置（对话与语音）
- 角色名称：会替换 System Prompt 中的 `$name$` 变量
- 上下文条数：参与本地 LLM 对话的历史消息条数（建议 1.5B≈12，7B≈24）
- 最大生成长度：单次生成的最大 token 数（建议 1.5B≈192，7B≈384）
- 对话人设（System Prompt）：自定义角色语气与边界，支持 `$name$` 变量
- LLM 规模与风格：选择 1.5B/7B 以及 Original/Universal/Anime
- 离线 TTS：启用后自动朗读 AI 最终回复；可选择 TTS 模型、设定说话人 sid 与音量

4) 高级设置
- 抗锯齿 MSAA：调节渲染采样（2x/4x/8x）
- 模型显示屏幕：多屏环境可指定显示屏
- 定时任务：支持每天固定时间或按间隔触发的本地提醒

## 从源码构建（手动步骤概览）
参考以下手动流程（简版）：

1) 获取依赖
- 安装 Git、CMake、VS 2022 Build Tools（含 VC 工具链）
- 安装 Miniconda 并创建环境：`conda create -y -n qt6env -c conda-forge qt6-main ninja`

2) 拉取与配置
- 克隆仓库：`git clone https://github.com/linhanmo/desktop-pet-with-llms.git`
- 准备 Qt6：找到 `Qt6_DIR` 与 `CMAKE_PREFIX_PATH`（通常在 `envs/qt6env/Library/lib/cmake/Qt6` 与 `envs/qt6env/Library`）

3) CMake 构建
- 生成：`cmake -S Pet -B Pet/build -G "Visual Studio 17 2022" -A x64 -DQt6_DIR=... -DCMAKE_PREFIX_PATH=...`
- 编译：`cmake --build Pet/build --config Release -j 8`
- 部署：使用 `windeployqt6.exe` 收集运行所需 DLL 到可执行目录

4) 放置资源
- `cubism.zip` 解压到 `Pet/sdk/` 下，确保最终为 `Pet/sdk/cubism/`
- `models.zip`、`voice_deps.zip.*`、`llm.zip.*` 解压/放置到 `Pet/res/` 下（详见“资源与目录约定”）

## 常见问题
- 运行后无对话/回复很短
  - 确认 `res/bin` 中存在 `llama-cli.exe/llama.exe`，并在“AI 设置”中选择合适规模与风格
  - 放入匹配的 gguf 模型至 `res/llm/1.5B` 或 `res/llm/7B`；或设置环境变量 `LLM_MODEL` 指定路径
- 无法发声/无可选 TTS 模型
  - 确认 `res/voice_deps/models` 下存在模型子目录，且包含模型/配置文件；启用“离线 TTS”
  - 多说话人模型需正确的说话人映射文件（如 speakers.txt 或 JSON 映射）
- 合并/解压失败
  - 确认所有分卷已完整下载并位于同一目录；用 7-Zip 从 `.001` 开始解压

## 示例命令（速查）
- 解压 build-msvc 分卷

```cmd
7z x build-msvc.zip.001
```

- 解压可选资源（分卷从 .001 开始解压）

```cmd
7z x voice_deps.zip.001
7z x llm.zip.001
7z x models.zip
```

- 将资源解压到正确位置（示例：在仓库 Pet 目录）

```powershell
Expand-Archive -LiteralPath .\assets\cubism.zip -DestinationPath .\sdk -Force
Rename-Item -Path .\sdk\cubism* -NewName cubism -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath .\assets\models.zip -DestinationPath .\res -Force
7z x .\assets\voice_deps.zip.001 -o.\res -y
New-Item -ItemType Directory -Force -Path .\res\llm | Out-Null
7z x .\assets\llm.zip.001 -o.\res\llm -y
```

- 指定 LLM 运行器与模型（环境变量覆盖）

```powershell
$env:LLAMA_RUNNER = "E:\XiaoMo\release\res\bin\llama-cli.exe"
$env:LLM_MODEL    = "E:\XiaoMo\release\res\llm\1.5B\your-model.gguf"
```

- llama-cli 快速自检（在 `release/` 目录执行）

```powershell
.\res\bin\llama-cli.exe -m .\res\llm\1.5B\your-model.gguf -p "你好" --simple-io -n 64
```

## 配置与日志路径
- 配置目录（Windows）：`%APPDATA%\IAIAYN\XiaoMo\Configs\config.json`
- 本地数据目录（Windows）：`%LOCALAPPDATA%\IAIAYN\XiaoMo`
- 启动日志：`%LOCALAPPDATA%\...\logs\startup.log`

## 许可证与致谢
- 本项目使用的第三方组件与模型版权归其各自所有者所有。Live2D 模型和语音模型请遵循其对应授权协议。

## 开发者信息与测试人员致谢
- 开发者：Mo
- 测试人员：guos7898-alt , xpresent-10

万分感谢所有测试人员的反馈与建议，他们帮助我全面优化了项目的核心性能与运行稳定性，打磨并提升了产品全链路的使用体验，更精准排查定位了多处潜在的程序缺陷与风险隐患，为项目的顺利落地与长期平稳运行筑牢了坚实根基。

百度网盘链接: https://pan.baidu.com/s/13QY8_rEd1pWLyFOL9JsFVQ?pwd=8888 提取码: 8888
