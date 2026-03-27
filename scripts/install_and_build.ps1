param(
  [string]$InstallRoot = "$env:USERPROFILE\XiaoMo",
  [ValidateSet('Portable','Full','Source')][string]$Mode = 'Portable',
  [string]$RepoUrl = "https://github.com/linhanmo/desktop-pet-with-llms.git",
  [string]$ReleaseTag = "v1.0.0-assets",
  [switch]$RunApp
)
$ErrorActionPreference = "Stop"
function Exec($c){Write-Host $c; & powershell -NoLogo -NoProfile -Command $c}
function Need($name,$test,$install){if(-not (& $test)){Write-Host "install:$name"; & $install}}
function Test-Cmd($n){$p=(Get-Command $n -ErrorAction SilentlyContinue);$null -ne $p}
function WingetInstall($id,$override){$cmd="winget install --id `"$id`" --silent --accept-package-agreements --accept-source-agreements";if($override){$cmd+=" --override `"$override`""};Exec $cmd}
function Fetch($url,$to){Write-Host "GET $url -> $to"; Invoke-WebRequest -Uri $url -OutFile $to -UseBasicParsing}
function Join-Parts($pattern,$to){
  $files = Get-ChildItem -File -Filter $pattern | Sort-Object Name
  if(-not $files){throw "no parts matched: $pattern"}
  $bytes = $files | Get-Content -Encoding Byte -ReadCount 0
  $bytes | Set-Content -Encoding Byte -Path $to
}

$AssetBase = "https://github.com/linhanmo/desktop-pet-with-llms/releases/download/$ReleaseTag"
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

if($Mode -eq 'Portable'){
  $zip = Join-Path $InstallRoot "XiaoMo_Portable.zip"
  Fetch "$AssetBase/XiaoMo_Portable.zip" $zip
  $outDir = Join-Path $InstallRoot "portable"
  if(Test-Path $outDir){Remove-Item -Recurse -Force $outDir}
  Expand-Archive -LiteralPath $zip -DestinationPath $outDir -Force
  $exe = Get-ChildItem -Path $outDir -Recurse -Filter "XiaoMo.exe" | Select-Object -First 1
  if($RunApp -and $exe){ Start-Process -FilePath $exe.FullName }
  Write-Host "done:$outDir"
  exit 0
}

if($Mode -eq 'Full'){
  $work = Join-Path $InstallRoot "full"
  New-Item -ItemType Directory -Force -Path $work | Out-Null
  Push-Location $work
  try{
    $parts = @("XiaoMo-windows.zip.part001","XiaoMo-windows.zip.part002","XiaoMo-windows.zip.part003")
    foreach($p in $parts){ Fetch "$AssetBase/$p" (Join-Path $work $p) }
    Join-Parts "XiaoMo-windows.zip.part*" (Join-Path $work "XiaoMo-windows.zip")
    $outDir = Join-Path $InstallRoot "XiaoMo"
    if(Test-Path $outDir){Remove-Item -Recurse -Force $outDir}
    Expand-Archive -LiteralPath (Join-Path $work "XiaoMo-windows.zip") -DestinationPath $outDir -Force
    $exe = Join-Path $outDir "XiaoMo.exe"
    if(-not (Test-Path $exe)){
      $exe = (Get-ChildItem -Path $outDir -Recurse -Filter "XiaoMo.exe" | Select-Object -First 1)?.FullName
    }
    if($RunApp -and $exe){ Start-Process -FilePath $exe }
    Write-Host "done:$outDir"
    exit 0
  } finally {
    Pop-Location
  }
}

# Mode = Source （从源码构建 + 下载必要资源）
if(!(Test-Cmd winget)){throw "winget not found"}
Need "Git" {Test-Cmd git} {WingetInstall "Git.Git" $null}
Need "CMake" {Test-Cmd cmake} {WingetInstall "Kitware.CMake" $null}
Need "7zip" {(Test-Cmd 7z) -or (Test-Path "C:\Program Files\7-Zip\7z.exe")} {WingetInstall "7zip.7zip" $null}
Need "VS Build Tools" {(Test-Path "C:\Program Files\Microsoft Visual Studio\2022\BuildTools")} {WingetInstall "Microsoft.VisualStudio.2022.BuildTools" "--quiet --wait --norestart --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"}
if(!(Test-Cmd conda)){ WingetInstall "Anaconda.Miniconda3" $null }
$CondaExe = (Get-Command conda -ErrorAction SilentlyContinue)?.Source
if(-not $CondaExe){ $CondaExe = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" }
if(-not (Test-Path $CondaExe)){ throw "conda not found" }
& $CondaExe env list | Out-Null
& $CondaExe create -y -n qt6env -c conda-forge qt6-main ninja | Out-Null
$CondaBase = (& $CondaExe info --base).Trim()
$EnvRoot = Join-Path $CondaBase "envs\qt6env"
$QtLibRoot = Join-Path $EnvRoot "Library"
$QtDir = Join-Path $QtLibRoot "lib\cmake\Qt6"
if(-not (Test-Path $QtDir)){ throw "Qt6 not found in env" }

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$SrcDir = Join-Path $InstallRoot "src"
if(Test-Path $SrcDir){ Remove-Item -Recurse -Force $SrcDir }
git clone --depth 1 $RepoUrl $SrcDir | Out-Null
$PetDir = Join-Path $SrcDir "Pet"
Set-Location $PetDir

# 下载运行资源（按需）：
$AssetsOut = Join-Path $InstallRoot "assets"
New-Item -ItemType Directory -Force -Path $AssetsOut | Out-Null
Fetch "$AssetBase/cubism.zip" (Join-Path $AssetsOut "cubism.zip")
Fetch "$AssetBase/models.zip" (Join-Path $AssetsOut "models.zip")
Fetch "$AssetBase/voice_deps.zip.part001" (Join-Path $AssetsOut "voice_deps.zip.part001")
Fetch "$AssetBase/voice_deps.zip.part002" (Join-Path $AssetsOut "voice_deps.zip.part002")
Join-Parts "voice_deps.zip.part*" (Join-Path $AssetsOut "voice_deps.zip")
# 小体量 LLM（可选；7B original 请手动下载以节省流量）
foreach($f in @(
  "llm-1.5B-anime.zip","llm-1.5B-original.zip","llm-1.5B-universal.zip",
  "llm-7B-anime.zip","llm-7B-universal.zip"
)){
  try{ Fetch "$AssetBase/$f" (Join-Path $AssetsOut $f) } catch { Write-Host "skip:$f" }
}

Expand-Archive -LiteralPath (Join-Path $AssetsOut "cubism.zip") -DestinationPath (Join-Path $PetDir "sdk") -Force
Rename-Item -Path (Join-Path $PetDir "sdk\cubism*") -NewName "cubism" -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath (Join-Path $AssetsOut "models.zip") -DestinationPath (Join-Path $PetDir "res") -Force
Expand-Archive -LiteralPath (Join-Path $AssetsOut "voice_deps.zip") -DestinationPath (Join-Path $PetDir "res") -Force
foreach($f in Get-ChildItem $AssetsOut -Filter "llm-*.zip"){
  Expand-Archive -LiteralPath $f.FullName -DestinationPath (Join-Path $PetDir "res\llm") -Force
}

$BuildDir = Join-Path $PetDir "build"
if(Test-Path $BuildDir){ Remove-Item -Recurse -Force $BuildDir }
cmake -S $PetDir -B $BuildDir -G "Visual Studio 17 2022" -A x64 -DQt6_DIR="$QtDir" -DCMAKE_PREFIX_PATH="$QtLibRoot" | Out-Null
cmake --build $BuildDir --config Release -j 8 | Out-Null
$BinDir = Join-Path $BuildDir "Release"
$Windeploy = Join-Path $QtLibRoot "bin\windeployqt6.exe"
if(Test-Path $Windeploy){
  & $Windeploy --release --compiler-runtime --force --dir $BinDir (Join-Path $BinDir "XiaoMo.exe") | Out-Null
}
if($RunApp){ Start-Process -FilePath (Join-Path $BinDir "XiaoMo.exe") }
Write-Host "done:$BinDir"
