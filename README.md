# XiaoMo 桌面宠物（离线本地 LLM + 语音合成）

一款基于 Qt + Live2D 的桌面宠物应用，支持本地离线 LLM 对话与离线 TTS（文字转语音），提供托盘菜单与全局快捷键，开箱即用或通过脚本一键部署/构建。


## 功能概览
- Live2D 看板娘：支持呼吸、眨眼、视线跟随、物理模拟（头发/衣物摆动），可设置屏幕显示、透明背景、全局置顶等。
- 本地离线 LLM 对话：内置 llama.cpp CLI 调用，支持 1.5B/7B 规模，风格可选 Original/Universal/Anime，System Prompt 可自定义。
- 离线 TTS 播报：集成 sherpa-onnx 生态的多种离线语音模型，支持多说话人（sid）、音量调节。
- 对话与气泡：角色旁显示对话气泡，支持不同气泡主题；聊天窗口支持清空历史、上下文条数与最大生成长度可调。
- 本地提醒：内置定时任务（每天固定时间或按间隔）弹出提示并可驱动动作组。
- 多语言与主题：支持简体中文、英文与主题切换；配置与对话历史持久化存储。


## 环境要求（运行）
- Windows 10/11（x64）。
- 若使用“便携包”或“全量包”，无需额外安装运行库（已包含必要 DLL）。
- 若用 MSIX 最小包，需启用开发者模式或使用受信任证书签名。


## 环境要求（从源码构建）
- Git、CMake、Visual Studio 2022 Build Tools（含 VC 工具链）。
- Miniconda（用于获取 Qt6）：脚本会创建 qt6env 并安装 qt6-main、ninja。
- Windows PowerShell + winget（脚本会自动安装缺失依赖）。


## 获取与运行
你可以选择三种方式部署/运行：

1) 便携包（推荐上手）
- 下载 Release 中的 XiaoMo_Portable.zip
- 解压后直接运行 XiaoMo.exe
- 如需本地 LLM 或离线语音，请另外下载对应资源包并解压到 res 子目录

2) 全量包（包含全部离线资源）
- 下载 XiaoMo-windows.zip.part001 ~ XiaoMo-windows.zip.part00N 全部分片
- 在同一目录合并：
  - Windows: `copy /b XiaoMo-windows.zip.part001+...+XiaoMo-windows.zip.part00N XiaoMo-windows.zip`
- 解压 XiaoMo-windows.zip 后运行 XiaoMo.exe

3) MSIX 最小包（仅最小运行时）
- 下载 XiaoMo_Minimal.msix 并安装（需要开发者模式或签名）
- 将需要的离线资源解压到安装目录的 res 子目录（结构见下文“资源与目录约定”）


## 一键脚本（下载/合并/构建）
仓库提供 PowerShell 脚本 scripts/install_and_build.ps1：

- 便携包部署：
  - `powershell -ExecutionPolicy Bypass -File scripts\install_and_build.ps1 -Mode Portable`
- 全量包（脚本自动探测分片并合并）：
  - `powershell -ExecutionPolicy Bypass -File scripts\install_and_build.ps1 -Mode Full`
- 从源码构建（自动准备 Qt6 与资源、构建并可运行）：
  - `powershell -ExecutionPolicy Bypass -File scripts\install_and_build.ps1 -Mode Source`
- 可选参数：
  - `-InstallRoot` 指定安装/工作目录（默认：`%USERPROFILE%\XiaoMo`）
  - `-RunApp` 完成后自动运行应用

脚本要点：
- 启用 TLS1.2 下载，兼容更多网络环境。
- Full 模式自动遍历分片（part001..part020），合并无须手动数分片。
- Source 模式会下载 cubism、models、voice_deps 分片并解压到 Pet/res。


## 资源与目录约定
应用运行时查找资源的根目录为程序目录下的 `res`：
```
XiaoMo.exe
res/
  bin/            # 包含 llama-cli.exe/llama.exe 等运行器（用于本地 LLM）
  llm/            # 本地 LLM 模型（*.gguf），可分规模子目录 1.5B/7B
  models/         # Live2D 模型集合（每个子目录一个模型）
  voice_deps/     # 离线语音模型与依赖（sherpa-onnx 等）
  i18n/, icons/   # 语言与图标资源
```

Live2D 模型（res/models）：
- 每个模型放在独立文件夹，如 `res/models/<模型名>/`。
- 模型文件优先匹配 `*.model3.json`，若无则尝试 `*.model.json`，否则回退 `model3.json/model.json/index.json`。
- 默认模型根目录（设置中可改）：`文档\XiaoMo\Models`。首次运行会自动使用其中的第一个模型。

本地 LLM 模型（res/llm）：
- 将 gguf 文件放入 `res/llm/1.5B/` 或 `res/llm/7B/`（可不分文件夹，程序会自动匹配）。
- 风格匹配规则（文件名包含以下关键字之一将优先选择）：
  - Anime：包含 `anime` 或 `anime.q`
  - Original：包含 `original` 或 `llama`
  - Universal：包含 `universal`
- 模型规模在“AI 设置”选择（1.5B 更快、7B 更强）；未命中风格关键字时会使用目录下第一项。
- 支持环境变量覆盖：
  - `LLAMA_RUNNER` 指定 llama 运行器路径
  - `LLM_MODEL` 指定 gguf 模型文件路径

离线 TTS（res/voice_deps）：
- 语音模型目录：`res/voice_deps/models/<模型名>/`，程序会自动列出可用模型。
- 多说话人（sid）：若模型包含说话人映射（speaker_id_map 或 speakers.txt），可在“AI 设置”里调节 sid。
- sherpa-onnx 可执行目录自动检测于 `res/voice_deps/sherpa-onnx-*/bin`。


## 快捷键与托盘
- Ctrl+T：显示/隐藏聊天窗口
- Ctrl+H：显示/隐藏桌宠窗口
- Ctrl+S：打开设置窗口
- 托盘图标：右键菜单可快速打开聊天、设置、退出；切换显示/隐藏状态


## 使用指南（设置窗口）
设置窗口分“基础设置 / 模型设置 / AI 设置 / 高级设置”四个页签：

1) 基础设置
- 主题与语言：切换应用主题与显示语言。
- 窗口行为：全局置顶、透明背景、鼠标穿透（穿透开启后需 Alt+Tab 切回再关闭）。
- 输出气泡：选择角色旁边对话气泡样式（如 爱心/漫画）。

2) 模型设置（Live2D）
- 模型选择：从默认模型根目录中选择不同模型；可一键打开当前模型所在目录。
- 去除水印：为当前模型选择一个 `*.exp3.json` 表达式文件作为“水印表达式”，以便覆盖并去除水印效果（可随时取消）。
- 动作与效果：开关“自动呼吸/自动眨眼/视线跟踪/物理模拟”。

3) AI 设置（对话与语音）
- 角色名称：会替换 System Prompt 中的 `$name$` 变量。
- 上下文条数：参与本地 LLM 对话的历史消息条数（建议 1.5B≈12，7B≈24）。
- 最大生成长度：单次生成的最大 token 数（建议 1.5B≈192，7B≈384）。
- 对话人设（System Prompt）：自定义角色语气与边界，支持 `$name$` 变量。
- LLM 规模与风格：选择 1.5B/7B 以及 Original/Universal/Anime。
- 离线 TTS：启用后自动朗读 AI 最终回复；可选择 TTS 模型、设定说话人 sid 与音量。

4) 高级设置
- 抗锯齿 MSAA：调节渲染采样（2x/4x/8x）。
- 模型显示屏幕：多屏环境可指定显示屏。
- 定时任务：支持每天固定时间或按间隔触发的本地提醒。


## 从源码构建（手动步骤概览）
若不使用脚本，可参考以下手动流程（简版）：

1) 获取依赖
- 安装 Git、CMake、VS 2022 Build Tools（含 VC 工具链）。
- 安装 Miniconda 并创建环境：`conda create -y -n qt6env -c conda-forge qt6-main ninja`

2) 拉取与配置
- 克隆仓库：`git clone https://github.com/linhanmo/desktop-pet-with-llms.git`
- 准备 Qt6：找到 `Qt6_DIR` 与 `CMAKE_PREFIX_PATH`（通常在 `envs/qt6env/Library/lib/cmake/Qt6` 与 `envs/qt6env/Library`）。

3) CMake 构建
- 生成：`cmake -S Pet -B Pet/build -G "Visual Studio 17 2022" -A x64 -DQt6_DIR=... -DCMAKE_PREFIX_PATH=...`
- 编译：`cmake --build Pet/build --config Release -j 8`
- 部署：使用 `windeployqt6.exe` 收集运行所需 DLL 到可执行目录。

4) 放置资源
- 将 `cubism` SDK（解压后为 `sdk/cubism`）与 `models/`、`voice_deps/`、`llm/` 等解压/放置到 `Pet/res` 下（详见“资源与目录约定”）。


## 常见问题
- 运行后无对话/回复很短
  - 确认 `res/bin` 中存在 `llama-cli.exe/llama.exe`，并在“AI 设置”中选择合适规模与风格。
  - 放入匹配的 gguf 模型至 `res/llm/1.5B` 或 `res/llm/7B`；或设置环境变量 `LLM_MODEL` 指定路径。
- 无法发声/无可选 TTS 模型
  - 确认 `res/voice_deps/models` 下存在模型子目录，且包含模型/配置文件；启用“离线 TTS”。
  - 多说话人模型需正确的说话人映射文件（如 speakers.txt 或 JSON 映射）。
- 合并/解压失败
  - 确认所有分片已经完整下载并位于同一目录；合并命令按顺序拼接；压缩包解压到对应目录。
- 脚本执行被阻止
  - 以管理员打开 PowerShell 或使用 `-ExecutionPolicy Bypass` 临时放行。
- MSIX 安装失败
  - 启用开发者模式或使用受信任证书对包进行签名。


## 示例命令（速查）

- 临时放行脚本执行（仅当前 PowerShell 会话）

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
```

- 脚本一键部署/构建

```powershell
# 便携包到自定义目录并自动运行
powershell -ExecutionPolicy Bypass -File scripts\install_and_build.ps1 -Mode Portable -InstallRoot "D:\XiaoMo" -RunApp

# 全量包（自动探测并合并分片）
powershell -ExecutionPolicy Bypass -File scripts\install_and_build.ps1 -Mode Full -InstallRoot "D:\XiaoMo"

# 从源码构建（自动准备 Qt6 与资源）
powershell -ExecutionPolicy Bypass -File scripts\install_and_build.ps1 -Mode Source -InstallRoot "D:\XiaoMo" -RunApp
```

- 分卷合并（手动）

```cmd
:: 合并全量包
copy /b XiaoMo-windows.zip.part001+XiaoMo-windows.zip.part002+...+XiaoMo-windows.zip.part00N XiaoMo-windows.zip

:: 合并离线语音依赖
copy /b voice_deps.zip.part001+voice_deps.zip.part002 voice_deps.zip

:: 合并 LLM 7B original（如有）
copy /b llm-7B-original.zip.part001+llm-7B-original.zip.part002+llm-7B-original.zip.part003 llm-7B-original.zip
```

- 将资源解压到正确位置（手动）

```powershell
# 假设当前在仓库 Pet 目录
Expand-Archive -LiteralPath .\assets\cubism.zip -DestinationPath .\sdk -Force
Rename-Item -Path .\sdk\cubism* -NewName cubism -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath .\assets\models.zip -DestinationPath .\res -Force
Expand-Archive -LiteralPath .\assets\voice_deps.zip -DestinationPath .\res -Force
New-Item -ItemType Directory -Force -Path .\res\llm | Out-Null
Get-ChildItem .\assets -Filter "llm-*.zip" | % { Expand-Archive -LiteralPath $_.FullName -DestinationPath .\res\llm -Force }
```

- 指定 LLM 运行器与模型（环境变量覆盖）

```powershell
$env:LLAMA_RUNNER = "E:\XiaoMo\portable\res\bin\llama-cli.exe"
$env:LLM_MODEL    = "E:\XiaoMo\portable\res\llm\1.5B\your-model.gguf"
# 运行 XiaoMo.exe 后将使用以上设置；关闭 PowerShell 该设置失效
```

- llama-cli 快速自检

```powershell
.\res\bin\llama-cli.exe -m .\res\llm\1.5B\your-model.gguf -p "你好" --simple-io -n 64
```

- 手动安装构建依赖（可选，脚本会自动安装）

```powershell
winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
winget install --id Kitware.CMake -e --accept-package-agreements --accept-source-agreements
winget install --id 7zip.7zip -e --accept-package-agreements --accept-source-agreements
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
```

- 准备 Qt6 的 Conda 环境（可与脚本一致）

```powershell
conda create -y -n qt6env -c conda-forge qt6-main ninja
```


## 配置与日志路径
- 配置目录（Windows）：`%APPDATA%\IAIAYN\XiaoMo\Configs\config.json`
- 本地数据目录（Windows）：`%LOCALAPPDATA%\IAIAYN\XiaoMo`
- 启动日志：`%LOCALAPPDATA%\... \logs\startup.log`


## 许可证与致谢
- 本项目使用的第三方组件与模型版权归其各自所有者所有。Live2D 模型和语音模型请遵循其对应授权协议。
