# TGAI Engine 构建脚本 (PowerShell)
# 用法:
#   .\build.ps1              # 编译当前平台
#   .\build.ps1 -Target android   # 交叉编译 Android ARM64
#   .\build.ps1 -Target windows   # 编译 Windows x64
#   .\build.ps1 -Target android-armv7  # Android ARM32

param(
    [ValidateSet('auto', 'windows', 'linux', 'android', 'android-armv7', 'macos', 'ios')]
    [string]$Target = 'auto',

    [int]$OptLevel = 3  # 0=无优化, 3=最大优化
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "=== TGAI Engine Build ===" -ForegroundColor Cyan
Write-Host "Target: $Target"
Write-Host "OptLevel: -O$OptLevel"
Write-Host ""

# ─── 检测目标平台 ────────────────────────────────
if ($Target -eq 'auto') {
    if ($IsWindows -or $env:OS -eq 'Windows_NT') { $Target = 'windows' }
    elseif ($IsLinux) { $Target = 'linux' }
    elseif ($IsMacOS) { $Target = 'macos' }
    else { throw "无法检测平台, 请指定 -Target" }
}

# ─── 编译器选择 ──────────────────────────────────
function Find-Compiler($expected) {
    $cmd = Get-Command $expected -ErrorAction SilentlyContinue
    if ($cmd) { return $expected }
    return $null
}

$cc = $null
$extraFlags = @()
$outExt = ''
$crossCompile = $false
$isTcc = $false

# ─── 优先找 TCC (内置) ─────────────────────────
$tccLocal = Join-Path $ScriptDir '..\tcc\tcc\tcc.exe'
if (Test-Path $tccLocal) {
    $cc = $tccLocal
    $isTcc = $true
    Write-Host "Found: TCC (bundled) at $cc"
}

# ─── 找更好的编译器 ─────────────────────────
if (-not $cc) {

switch ($Target) {
    'windows' {
        $cc = Find-Compiler 'clang' ; if (-not $cc) { $cc = Find-Compiler 'gcc' }
        if (-not $cc) { $cc = Find-Compiler 'cl' }
        $outExt = '.dll'
        $extraFlags = @('-shared', '-O2')
    }
    'linux' {
        $cc = Find-Compiler 'clang' ; if (-not $cc) { $cc = Find-Compiler 'gcc' }
        $outExt = '.so'
        $extraFlags = @('-shared', '-fPIC', "-O$OptLevel", '-lm')
    }
    'macos' {
        $cc = Find-Compiler 'clang'
        $outExt = '.dylib'
        $extraFlags = @('-shared', '-fPIC', "-O$OptLevel", '-lm')
    }
    'android' {
        # 需要 ANDROID_NDK 环境变量
        $ndk = $env:ANDROID_NDK
        if (-not $ndk) { $ndk = $env:ANDROID_NDK_HOME }
        if (-not $ndk) { throw "需要设置 ANDROID_NDK 环境变量" }
        $toolchain = Join-Path $ndk "toolchains/llvm/prebuilt"
        $host = if ($IsWindows -or $env:OS -eq 'Windows_NT') { 'windows-x86_64' }
                elseif ($IsLinux) { 'linux-x86_64' }
                else { 'darwin-x86_64' }
        $cc = Join-Path $toolchain "$host/bin/aarch64-linux-android24-clang"
        $outExt = '.so'
        $extraFlags = @('-shared', '-fPIC', "-O$OptLevel", '-lm',
                        '-target', 'aarch64-linux-android24')
        $crossCompile = $true
    }
    'android-armv7' {
        $ndk = $env:ANDROID_NDK
        if (-not $ndk) { $ndk = $env:ANDROID_NDK_HOME }
        if (-not $ndk) { throw "需要设置 ANDROID_NDK 环境变量" }
        $toolchain = Join-Path $ndk "toolchains/llvm/prebuilt"
        $host = if ($IsWindows -or $env:OS -eq 'Windows_NT') { 'windows-x86_64' }
                elseif ($IsLinux) { 'linux-x86_64' }
                else { 'darwin-x86_64' }
        $cc = Join-Path $toolchain "$host/bin/armv7a-linux-androideabi24-clang"
        $outExt = '.so'
        $extraFlags = @('-shared', '-fPIC', "-O$OptLevel", '-lm',
                        '-target', 'armv7a-linux-androideabi24',
                        '-mfloat-abi=softfp', '-mfpu=neon')
        $crossCompile = $true
    }
    'ios' {
        $cc = Find-Compiler 'clang'
        if (-not $cc) { throw "iOS 需要 clang" }
        $outExt = '.dylib'
        $extraFlags = @('-shared', '-fPIC', "-O$OptLevel", '-lm',
                        '-target', 'arm64-apple-ios14.0',
                        '-isysroot', '/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk')
        $crossCompile = $true
    }
}

if (-not $cc) { throw "找不到合适的编译器" }

Write-Host "Compiler: $cc"
if ($isTcc) { Write-Host "⚠ TCC 不支持 SIMD，速度较慢。建议安装 MinGW-w64 或 clang。" -ForegroundColor Yellow }
Write-Host ""

# ─── 编译 ────────────────────────────────────────
$outName = "libtgai_engine$outExt"
$outPath = Join-Path $ScriptDir "build\$outName"
$buildDir = Join-Path $ScriptDir "build"
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

$src = Join-Path $ScriptDir '..\src\tgai_engine.c'

# 判断编译器家族
$isMsvc = $false
if ($cc -eq 'cl' -or $cc -like '*cl.exe') { $isMsvc = $true }
elseif ($cc -like '*clang-cl*') { $isMsvc = $true }

if ($isTcc) {
    # TCC: 最简编译 (无 SIMD，速度慢但够用)
    $flags = @('-shared', '-o', $outPath, $src, "-I$ScriptDir\..\src")
    Write-Host "TCC compile: $cc $flags" -ForegroundColor DarkGray
    & $cc @flags
    if ($LASTEXITCODE -ne 0) { throw "TCC 编译失败" }
} elseif ($isMsvc) {
    # MSVC (cl.exe / clang-cl.exe) 参数风格
    $flags = @(
        '/LD',                      # 生成 DLL
        "/O$($OptLevel)",           # 优化级别 (O2 默认)
        '/Oy',                      # 省略帧指针
        '/fp:fast',                 # 快速数学
        '/std:c11',
        '/W3',
        '/D_CRT_SECURE_NO_WARNINGS',
        "/Fo:$buildDir\\",          # 临时 obj 目录
        "/Fe:$outPath",             # 输出文件
        $src
    )
    # clang-cl 支持 AVX2
    if ($Target -eq 'windows' -and $cc -like '*clang-cl*') {
        $flags += @('-mavx2', '-mfma')
    }
} else {
    # GCC / Clang 参数风格
    $flags = @($src, '-o', $outPath) + $extraFlags + @(
        '-std=c99',
        '-Wall',
        '-Wextra',
        "-O$OptLevel",
        '-ffast-math',          # 快速数学 (允许 FP 重新关联)
        '-funroll-loops',       # 循环展开
        '-D_POSIX_C_SOURCE=200809L',
    )

    # SIMD 标志
    if ($Target -in @('android', 'android-armv7', 'macos', 'ios')) {
        $flags += @('-mfpu=neon-fp16')  # ARM NEON
    } elseif ($Target -eq 'windows' -or $Target -eq 'linux') {
        $flags += @('-mavx2', '-mfma', '-mbmi2')  # x86 AVX2
    }
}

Write-Host "Compile flags:"
Write-Host "  $flags" -ForegroundColor DarkGray
Write-Host ""

& $cc @flags
if ($LASTEXITCODE -ne 0) { throw "编译失败" }

# ─── 验证 ────────────────────────────────────────
$size = (Get-Item $outPath).Length
Write-Host ""
Write-Host "✓ 编译成功!" -ForegroundColor Green
Write-Host "  输出: $outPath"
Write-Host "  大小: $($size / 1024) KB"
Write-Host ""

# 显示构建产物
if ($crossCompile) {
    Write-Host "交叉编译产物, 不能在本机运行, 需部署到目标设备。" -ForegroundColor Yellow
} else {
    Write-Host "本机可直接加载。Flutter 中:"
    Write-Host "  - Windows: 复制到 tg_chat\windows\flutter\"
    Write-Host "  - Linux: 复制到 tg_chat\linux\flutter\lib\"
}
