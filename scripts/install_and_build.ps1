param(
  [string]$InstallRoot = "$env:USERPROFILE\XiaoMo",
  [string]$RepoUrl = "https://github.com/linhanmo/desktop-pet-with-llms.git",
  [string]$ReleaseTag = "v1.0.0-assets",
  [switch]$RunApp
)
$ErrorActionPreference = "Stop"
function Exec($c){Write-Host $c; & powershell -NoLogo -NoProfile -Command $c}
function Need($name,$test,$install){if(-not (& $test)){Write-Host "install:$name"; & $install}}
function Test-Cmd($n){$p=(Get-Command $n -ErrorAction SilentlyContinue);$null -ne $p}
function WingetInstall($id,$override){$cmd="winget install --id `"$id`" --silent --accept-package-agreements --accept-source-agreements";if($override){$cmd+=" --override `"$override`""};Exec $cmd}
if(!(Test-Cmd winget)){throw "winget not found"}
Need "Git" {Test-Cmd git} {WingetInstall "Git.Git" $null}
Need "CMake" {Test-Cmd cmake} {WingetInstall "Kitware.CMake" $null}
Need "7zip" {(Test-Cmd 7z) -or (Test-Path "C:\Program Files\7-Zip\7z.exe")} {WingetInstall "7zip.7zip" $null}
Need "VS Build Tools" {(Test-Path "C:\Program Files\Microsoft Visual Studio\2022\BuildTools")} {WingetInstall "Microsoft.VisualStudio.2022.BuildTools" "--quiet --wait --norestart --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"}
if(!(Test-Cmd conda)){
  WingetInstall "Anaconda.Miniconda3" $null
}
$CondaExe = (Get-Command conda -ErrorAction SilentlyContinue)?.Source
if(-not $CondaExe){
  $CondaExe = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe"
}
if(-not (Test-Path $CondaExe)){throw "conda not found"}
& $CondaExe env list | Out-Null
& $CondaExe create -y -n qt6env -c conda-forge qt6-main ninja | Out-Null
$CondaBase = (& $CondaExe info --base).Trim()
$EnvRoot = Join-Path $CondaBase "envs\qt6env"
$QtLibRoot = Join-Path $EnvRoot "Library"
$QtDir = Join-Path $QtLibRoot "lib\cmake\Qt6"
if(-not (Test-Path $QtDir)){throw "Qt6 not found in env"}
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$SrcDir = Join-Path $InstallRoot "src"
if(Test-Path $SrcDir){Remove-Item -Recurse -Force $SrcDir}
git clone --depth 1 $RepoUrl $SrcDir | Out-Null
$PetDir = Join-Path $SrcDir "Pet"
Set-Location $PetDir
New-Item -ItemType Directory -Force -Path (Join-Path $InstallRoot "assets") | Out-Null
$AssetBase = "https://github.com/linhanmo/desktop-pet-with-llms/releases/download/$ReleaseTag"
function Fetch($name,$to){
  $url = "$AssetBase/$name"
  Invoke-WebRequest -Uri $url -OutFile $to -UseBasicParsing
}
$AssetsOut = Join-Path $InstallRoot "assets"
$parts = 1..6 | ForEach-Object { "res_archive.zip.{0:D3}" -f $_ }
foreach($p in $parts){Fetch $p (Join-Path $AssetsOut $p)}
Fetch "cubism.zip" (Join-Path $AssetsOut "cubism.zip")
Set-Location $AssetsOut
Get-ChildItem -Filter "res_archive.zip.*" | Sort-Object Name | Get-Content -Encoding Byte -ReadCount 0 | Set-Content -Encoding Byte (Join-Path $AssetsOut "res_archive.zip")
Expand-Archive -LiteralPath (Join-Path $AssetsOut "res_archive.zip") -DestinationPath $SrcDir -Force
Expand-Archive -LiteralPath (Join-Path $AssetsOut "cubism.zip") -DestinationPath (Join-Path $PetDir "sdk") -Force
Rename-Item -Path (Join-Path $PetDir "sdk\cubism*") -NewName "cubism" -ErrorAction SilentlyContinue
Set-Location $PetDir
$BuildDir = Join-Path $PetDir "build"
if(Test-Path $BuildDir){Remove-Item -Recurse -Force $BuildDir}
cmake -S $PetDir -B $BuildDir -G "Visual Studio 17 2022" -A x64 -DQt6_DIR="$QtDir" -DCMAKE_PREFIX_PATH="$QtLibRoot" | Out-Null
cmake --build $BuildDir --config Release -j 8 | Out-Null
$BinDir = Join-Path $BuildDir "Release"
$Windeploy = Join-Path $QtLibRoot "bin\windeployqt6.exe"
if(Test-Path $Windeploy){
  & $Windeploy --release --compiler-runtime --force --dir $BinDir (Join-Path $BinDir "XiaoMo.exe") | Out-Null
}
if($RunApp){Start-Process -FilePath (Join-Path $BinDir "XiaoMo.exe")}
Write-Host "done:$BinDir"
