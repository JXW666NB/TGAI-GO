# TGAI Engine 构建脚本 (PowerShell)
# 用法:
#   .\build.ps1              # 编译当前平台 (纯文本 LLM)
#   .\build.ps1 -MultiFile  # 编译当前平台 + 多模态 (DiT/VAE/AudioCodec/Router)
#   .\build.ps1 -Target android   # 交叉编译 Android ARM64
#   .\build.ps1 -Target windows   # 编译 Windows x64
#   .\build.ps1 -Target android-armv7  # Android ARM32
#   .\build.ps1 -MultiFile -Target windows  # 多模态 + Windows

param(
    [ValidateSet('auto', 'windows', 'linux', 'android', 'android-armv7', 'macos', 'ios')]
    [string]$Target = 'auto',

    [int]$OptLevel = 3,  # 0=无优化, 3=最大优化

    # 启用多模态模块 (mm/): 加入 DiT/VAE/AudioCodec/Scheduler/Router, 支持 .TG v3
    [switch]$MultiFile = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "=== TGAI Engine Build ===" -ForegroundColor Cyan
Write-Host "Target: $Target"
Write-Host "OptLevel: -O$OptLevel"
Write-Host "MultiFile: $MultiFile"
Write-Host ""

# ─── 收集 mm 模块源文件 (当 -MultiFile 启用时) ─────
$mmSrc = @()
$mmDefines = @()
if ($MultiFile) {
    $mmDir = Join-Path $ScriptDir '..\src\mm'
    if (Test-Path $mmDir) {
        $mmSrc = @(Get-ChildItem -Path $mmDir -Filter '*.c' -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName } | Sort-Object)
        Write-Host "MultiFile 启用: 加入 $($mmSrc.Count) 个 mm 模块文件" -ForegroundColor Cyan
        foreach ($f in $mmSrc) { Write-Host "  - $(Split-Path -Leaf $f)" -ForegroundColor DarkGray }
        Write-Host ""
    } else {
        Write-Host "警告: -MultiFile 启用但 mm/ 目录不存在, 回退纯文本模式" -ForegroundColor Yellow
        $MultiFile = $false
    }
    $mmDefines = @('-DTG_ENABLE_MM')
}

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

# ─── TCC (内置, 兜底; 仅当无 clang/gcc/cl 时) ─────────────────────────
# clang/gcc/cl 支持 OpenMP + SIMD, 性能远优于 TCC; 优先用它们, TCC 仅兜底
$tccLocal = Join-Path $ScriptDir '..\tcc\tcc\tcc.exe'
if ((Test-Path $tccLocal) -and -not (Find-Compiler 'clang') -and -not (Find-Compiler 'gcc') -and -not (Find-Compiler 'cl')) {
    $cc = $tccLocal
    $isTcc = $true
    # TCC 在各 target 产物扩展名 (与下方 switch 保持一致, 否则 Windows 输出
    # 无 .dll 后缀, LoadLibrary 加载失败)
    switch ($Target) {
        'windows' { $outExt = '.dll' }
        'linux'   { $outExt = '.so' }
        'macos'   { $outExt = '.dylib' }
        'android' { $outExt = '.so' }
        'android-armv7' { $outExt = '.so' }
        'ios'     { $outExt = '.dylib' }
    }
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
}  # 闭合 if (-not $cc) (原版遗留: switch 的 } 后缺少此闭合)

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
    # 统一在下方 & $cc @flags 编译, 此处仅构造 flags
    # TCC 无 pthread.h, 强制单线程 (TG_SINGLE_THREAD 禁用 tgai_engine.c 的 pthread 路径)
    $flags = @('-shared', '-o', $outPath, $src) + $mmSrc + @("-I$ScriptDir\..\src", "-I$ScriptDir\..\src\mm") + $mmDefines + @('-DTG_SINGLE_THREAD')
} elseif ($isMsvc) {
    # MSVC (cl.exe / clang-cl.exe) 参数风格
    $flags = @(
        '/LD',                      # 生成 DLL
        # ★ MSVC 只有 /Od /O1 /O2 /Ox（没有 /O3，传 /O3 会 D9002 被忽略）；
        #   把 GCC 风格的 OptLevel 映射过来，否则等于没优化。
        $(if ($OptLevel -ge 2) { "/O2" } elseif ($OptLevel -eq 1) { "/O1" } else { "/Od" }),
        '/Oy',                      # 省略帧指针
        '/fp:fast',                 # 快速数学
        '/openmp',                  # 启用 OpenMP (DiT/VAE #pragma omp 并行, 多核加速文生图)
        # ★ 源码是 UTF-8（含大量中文注释/字符串），MSVC 默认按系统代码页(936)
        #   读取会把中文字符串切断 → error C2001/C2146。必须显式 /utf-8。
        '/utf-8',
        '/std:c11',
        '/W3',
        '/D_CRT_SECURE_NO_WARNINGS'
    ) + $mmDefines + @(
        "/Fo:$buildDir\\",          # 临时 obj 目录
        "/Fe:$outPath",             # 输出文件
        $src
    ) + $mmSrc
    # MSVC 需要额外 include 路径
    $flags += @("-I$ScriptDir\..\src", "-I$ScriptDir\..\src\mm")
    # ★ AVX2：clang-cl 用 -mavx2；MSVC 用 /arch:AVX2
    if ($Target -eq 'windows') {
        if ($cc -like '*clang-cl*') {
            $flags += @('-mavx2', '-mfma')
        } else {
            $flags += @('/arch:AVX2', '/fp:fast')
        }
    }
} else {
    # GCC / Clang 参数风格
    $flags = @($src) + $mmSrc + @('-o', $outPath) + $extraFlags + $mmDefines + @(
        '-std=c99',
        '-Wall',
        '-Wextra',
        "-O$OptLevel",
        '-ffast-math',          # 快速数学 (允许 FP 重新关联)
        '-funroll-loops',       # 循环展开
        '-D_POSIX_C_SOURCE=200809L',
        '-pthread',             # tgai_engine.c 用 pthread 多线程
        '-fopenmp',             # 启用 OpenMP (DiT/VAE #pragma omp 并行, 多核加速文生图)
        "-I$ScriptDir\..\src",
        "-I$ScriptDir\..\src\mm"
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

# ─── OpenMP 运行时依赖 (clang/gcc -fopenmp) ─────────────────────────
# clang/gcc 的 -fopenmp 生成依赖 libomp.dll/libgomp-1.dll 的 DLL;
# 复制到输出目录, 否则 ctypes 加载报 "找不到模块" (LoadLibrary 依赖解析)
if (-not $isTcc -and -not $isMsvc -and -not $crossCompile) {
    $ccResolved = (Get-Command $cc -ErrorAction SilentlyContinue).Source
    if (-not $ccResolved) { $ccResolved = $cc }
    $ccDir = Split-Path $ccResolved -Parent
    foreach ($ompDll in @('libomp.dll', 'libgomp-1.dll')) {
        $ompSrc = Join-Path $ccDir $ompDll
        if (Test-Path $ompSrc) {
            Copy-Item $ompSrc $buildDir -Force
            Write-Host "  复制 OpenMP 运行时: $ompDll" -ForegroundColor DarkGray
        }
    }
}

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
