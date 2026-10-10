﻿﻿﻿# TGAI Engine — Windows CUDA 编译脚本 (PowerShell)
# =================================================
# 用 nvcc + clang-cl 编译 GPU 版引擎: libtgai_engine_cuda.dll
#   - 纯文本 LLM 推理走 GPU (cuBLAS + 自定义 kernel)
#   - 输出 DLL 与 CPU 版 (libtgai_engine.dll) 并存, 互不覆盖
#   - 同时复制 CUDA 运行库 (cublas/cublasLt/cudart) 到输出目录,
#     编译后即使卸载 CUDA Toolkit, GPU DLL 仍可运行
#
# 用法:
#   .\build_cuda_win.ps1                                   # 自动探测工具链
#   .\build_cuda_win.ps1 -ComputeCapability sm_75          # 指定算力 (默认 sm_60)
#   .\build_cuda_win.ps1 -OutDir "D:\out"                  # 指定输出目录
#
# 前置要求 (Windows):
#   1. NVIDIA CUDA Toolkit (nvcc + cuBLAS), 建议 12.x:
#      https://developer.nvidia.com/cuda-downloads
#   2. LLVM/Clang (clang-cl), 建议 17+:
#      https://github.com/llvm/llvm-project/releases
#
# 注意: 本脚本不启用多模态 (纯文本 LLM GPU 加速)。
#       不定义 TG_ENABLE_MM, 与 Windows CPU 版 build.ps1 默认行为一致。

param(
    # 仓库根目录 (自动定位: 脚本上级的上级 = TGAI GO/)
    [string]$RepoRoot = "",

    # 输出目录 (默认 <RepoRoot>/cpu/build)
    [string]$OutDir = "",

    # CUDA 计算能力 (默认 sm_60 = 兼容 GTX 10 系及以上; CUDA 13+ 需 >= sm_70)
    [string]$ComputeCapability = "sm_60",

    # 主机编译器: auto=自动探测 clang-cl (LLVM); clang-cl=强制指定
    # 说明: Windows 上 nvcc 官方支持 clang-cl / MSVC 两类主机编译器。
    #       本脚本聚焦 clang-cl (LLVM 单装, 自带 lld-link, 无需 VS 环境)。
    #       MSVC 用户请手动在 VS Developer PowerShell 中运行并自行传参。
    [ValidateSet('auto', 'clang-cl')]
    [string]$HostCompiler = 'auto'
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# ── 定位仓库根 / 输出目录 ──────────────────────────
# 脚本在 <RepoRoot>/cuda/ 下: RepoRoot = 脚本目录的上级
if (-not $RepoRoot) { $RepoRoot = Split-Path $ScriptDir -Parent }
if (-not $OutDir)   { $OutDir = Join-Path $RepoRoot "cpu\build" }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  TGAI CUDA 引擎编译 (Windows)" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "RepoRoot: $RepoRoot"
Write-Host "OutDir:   $OutDir"
Write-Host "Arch:     $ComputeCapability"

# ── 检测 nvcc ──────────────────────────────────────
function Find-Nvcc {
    $cand = @()
    if ($env:CUDA_PATH) { $cand += Join-Path $env:CUDA_PATH "bin\nvcc.exe" }
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd) { $cand += $cmd.Source }
    foreach ($p in $cand) { if (Test-Path $p) { return $p } }
    # 常见安装路径兜底
    $verDir = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($verDir) {
        $p = Join-Path $verDir.FullName "bin\nvcc.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}
$nvcc = Find-Nvcc
if (-not $nvcc) {
    Write-Host "❌ 未找到 nvcc.exe. 请先安装 NVIDIA CUDA Toolkit:" -ForegroundColor Red
    Write-Host "   https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    exit 1
}
Write-Host "nvcc: $nvcc"

# ── nvcc 版本 → 默认算力下限 (CUDA 13+ 不再支持 Pascal sm_60/61) ──
try {
    $nvVer = (& $nvcc --version | Select-String 'release (\d+\.\d+)').Matches.Groups[1].Value
    $nvMajor = [int]($nvVer.Split('.')[0])
    if ($nvMajor -ge 13 -and $ComputeCapability -eq "sm_60") {
        Write-Host "⚠ CUDA $nvVer 已移除 Pascal 支持, 算力调整为 sm_70" -ForegroundColor Yellow
        $ComputeCapability = "sm_70"
    }
    Write-Host "CUDA:  $nvVer"
} catch { Write-Host "CUDA:  版本读取失败 (忽略)" }

# ── 检测主机编译器 (优先 MSVC cl.exe; clang-cl 兜底) ──
function Find-Compiler {
    # 1) MSVC cl.exe — nvcc 官方支持, 最稳。通过 vswhere 定位 VS/BuildTools。
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($installPath) {
            $msvcDir = Get-ChildItem (Join-Path $installPath "VC\Tools\MSVC") -Directory `
                -ErrorAction SilentlyContinue |
                Sort-Object { try { [version]$_.Name } catch { [version]"0.0" } } -Descending |
                Select-Object -First 1
            if ($msvcDir) {
                $cl = Join-Path $msvcDir.FullName "bin\Hostx64\x64\cl.exe"
                if (Test-Path $cl) {
                    return @{ Name='MSVC'; Exe=$cl; BinDir=(Split-Path $cl -Parent) }
                }
            }
        }
    }
    # 2) clang-cl (LLVM): 仍需 MSVC 头文件环境, 故仅作兜底
    $llvmDir = $null
    if ($env:LLVM_PATH) { $llvmDir = $env:LLVM_PATH }
    if (-not $llvmDir) {
        $p = "C:\Program Files\LLVM\bin"
        if (Test-Path $p) { $llvmDir = $p }
    }
    $cc = Get-Command clang-cl -ErrorAction SilentlyContinue
    if ($cc) { return @{ Name='clang-cl'; Exe=$cc.Source; BinDir=(Split-Path $cc.Source -Parent) } }
    if ($llvmDir -and (Test-Path (Join-Path $llvmDir "clang-cl.exe"))) {
        return @{ Name='clang-cl'; Exe=(Join-Path $llvmDir "clang-cl.exe"); BinDir=$llvmDir }
    }
    return $null
}
$compiler = Find-Compiler
if (-not $compiler) {
    Write-Host "❌ 未找到可用主机编译器 (MSVC cl.exe 或 clang-cl)." -ForegroundColor Red
    Write-Host "   安装 Visual Studio Build Tools (勾选 C++ 工具集):" -ForegroundColor Yellow
    Write-Host "   https://aka.ms/vs/17/release/vs_BuildTools.exe" -ForegroundColor Yellow
    exit 1
}
Write-Host "主机编译器: $($compiler.Name)  ($($compiler.Exe))"

# ── 构造编译命令 ────────────────────────────────────
$srcFile   = Join-Path $RepoRoot "src\tgai_engine.c"
$cuFile    = Join-Path $RepoRoot "cuda\tgai_cuda.cu"
$incDir    = Join-Path $RepoRoot "src"
$outDll    = Join-Path $OutDir "libtgai_engine_cuda.dll"

foreach ($f in @($srcFile, $cuFile, $incDir)) {
    if (-not (Test-Path $f)) {
        Write-Host "❌ 缺少源文件: $f" -ForegroundColor Red
        exit 1
    }
}

# Windows 需 -DTG_SINGLE_THREAD: tgai_engine.c 的线程池依赖 pthread,
# Windows 无 pthread.h; CPU 并行计算由 CUDA 接管, 单线程可接受
$define = "-DTG_SINGLE_THREAD"

# 主机编译器绑定目录
$ccbArgs = @("--compiler-bindir", $compiler.BinDir)

# 主机编译参数 (按编译器家族区分):
# - 源码为 UTF-8 无 BOM, 必须 /utf-8, 否则中文注释按 GBK 误读报 C2001
# - MSVC 无 -mavx2/-mfma, 用 /arch:AVX2; clang-cl 用 -mavx2 -mfma
if ($compiler.Name -eq 'MSVC') {
    $xccArgs = @("-Xcompiler", "/utf-8", "-Xcompiler", "/arch:AVX2")
} else {
    $xccArgs = @("-Xcompiler", "/utf-8", "-Xcompiler", "-mavx2", "-Xcompiler", "-mfma")
}

# 注意: PS5 对裸 token `-arch=$var` 不做变量展开 (实测原样传给 nvcc),
# 必须先拼成字符串变量再作为独立参数传入
$archArg = "-arch=$ComputeCapability"

Write-Host ""
Write-Host "编译中 (首次可能较慢, 请耐心)..." -ForegroundColor Cyan

& $nvcc @ccbArgs `
    -shared -O3 $archArg `
    @xccArgs `
    -I $incDir `
    $define `
    -o $outDll `
    $srcFile $cuFile `
    -lcublas

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 编译失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
    Write-Host "   常见原因:"
    Write-Host "   - 缺少 cuBLAS: 安装 CUDA Toolkit 时勾选 CUDA 组件"
    Write-Host "   - clang 版本过低: 建议 LLVM 17+"
    Write-Host "   - 算力不匹配: 用 -ComputeCapability 指定 (如 sm_75)"
    exit 1
}

# ── 复制 CUDA 运行库 (编译后卸载 Toolkit 也能跑) ──
$cudaBin = Join-Path (Split-Path $nvcc -Parent) "..\bin"
$runtimePatterns = @('cublas64_*.dll', 'cublasLt64_*.dll', 'cudart64_*.dll')
foreach ($pat in $runtimePatterns) {
    Get-ChildItem -Path $cudaBin -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName $OutDir -Force
        Write-Host "  + 运行库: $($_.Name)" -ForegroundColor DarkGray
    }
}

# ── 校验 ────────────────────────────────────────────
if (-not (Test-Path $outDll)) {
    Write-Host "❌ 输出 DLL 未生成" -ForegroundColor Red
    exit 1
}
$size = (Get-Item $outDll).Length / 1MB
Write-Host ""
Write-Host "✓ 编译成功!" -ForegroundColor Green
Write-Host "  DLL:   $outDll"
Write-Host "  大小:  $([math]::Round($size,1)) MB"
Write-Host ""
Write-Host "  TGAI Studio 会把该 DLL 放在 engine/ 目录并自动启用 GPU 加速"
Write-Host "  (需本机有 NVIDIA 显卡; 无卡自动回退 CPU, 不受影响)"
