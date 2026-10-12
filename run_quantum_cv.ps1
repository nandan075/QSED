<#
.SYNOPSIS
    Unified Runner for Quantum Computer Vision Models (QSED, QCSI, Pairwise Von Neumann Entropy).

.DESCRIPTION
    Executes all three quantum computer vision frameworks on high-resolution image pairs
    or across crisscross combinations using a single PowerShell command:
    1. QSED (Quantum SIFT/SURF + 4-Direction Rotation + Von Neumann Entropy Density Matrix)
    2. QCSI (Hybrid Quantum-Classical Image Similarity with RootSIFT 7-Qubit & DINOv2 ZZFeatureMap)
    3. Pairwise Von Neumann Entropy (Patch Quantum Pure States + Spectral Compatibility Matching + RANSAC)

.PARAMETER Img1
    Path to reference image 1.

.PARAMETER Img2
    Path to query image 2.

.PARAMETER Images
    Array of image paths for crisscross evaluation.

.PARAMETER Crisscross
    Switch to run all pairwise combinations among the supplied images.

.PARAMETER Detector
    Keypoint detector algorithm ('sift', 'orb', 'harris', 'qsed_gradient'). Default: 'sift'.

.PARAMETER MaxKps
    Maximum keypoints for quantum feature encoding. Default: 300.

.PARAMETER HighRes
    Preserve native high resolution without downscaling (up to 4096px). Default: $true.

.PARAMETER OutputDir
    Destination directory for dashboards, charts, and metrics JSON. Default: 'outputs\quantum_cv_unified'.

.PARAMETER AllCollege
    Convenience preset: Runs both the primary college pair and the 3-image crisscross set.

.EXAMPLE
    # Run primary pair on high resolution:
    .\run_quantum_cv.ps1 -Img1 "C:\college\IMG_20240825_121630.jpg" -Img2 "C:\college\IMG_20240825_121629.jpg" -Detector sift

.EXAMPLE
    # Run crisscross evaluation on 3 images:
    .\run_quantum_cv.ps1 -Crisscross -Images "C:\college\IMG_20240928_201329.jpg", "C:\college\IMG_20240928_201244.jpg", "C:\college\IMG_20240928_201211.jpg"

.EXAMPLE
    # Run the full suite:
    .\run_quantum_cv.ps1 -AllCollege
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [string]$Img1,

    [Parameter(Position = 1)]
    [string]$Img2,

    [Parameter()]
    [string[]]$Images,

    [Parameter()]
    [switch]$Crisscross,

    [Parameter()]
    [ValidateSet("sift", "orb", "harris", "qsed_gradient")]
    [string]$Detector = "sift",

    [Parameter()]
    [int]$MaxKps = 300,

    [Parameter()]
    [switch]$HighRes = $true,

    [Parameter()]
    [string]$OutputDir = "outputs\quantum_cv_unified",

    [Parameter()]
    [switch]$AllCollege
)

$ErrorActionPreference = "Stop"

# Banner
Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "       QUANTUM COMPUTER VISION: UNIFIED 3-MODEL BENCHMARK RUNNER           " -ForegroundColor Yellow
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " Models Loaded:" -ForegroundColor White
Write-Host "   [1] QSED                        - 7-Qubit SIFT Density Matrix & 4-Dir Rotation" -ForegroundColor Gray
Write-Host "   [2] QCSI                        - RootSIFT 7-Qubit + DINOv2 ZZFeatureMap Kernel" -ForegroundColor Gray
Write-Host "   [3] Pairwise Von Neumann Entropy- Relational Spectral Compatibility & RANSAC" -ForegroundColor Gray
Write-Host " Detector       : $Detector" -ForegroundColor White
Write-Host " Max Keypoints  : $MaxKps" -ForegroundColor White
Write-Host " High Resolution: $($HighRes.IsPresent -or $HighRes)" -ForegroundColor White
Write-Host " Output Dir     : $OutputDir" -ForegroundColor White
Write-Host "--------------------------------------------------------------------------" -ForegroundColor Cyan

# Locate Virtual Environment Python
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    # Fallback to system python
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PythonExe) {
        Write-Error "Could not locate python.exe in .venv or system PATH. Please ensure the virtual environment is available."
        exit 1
    }
}

Write-Host " Using Python   : $PythonExe" -ForegroundColor DarkGray

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$RunnerScript = Join-Path $ScriptDir "run_all_models.py"

# Handle -AllCollege preset
if ($AllCollege) {
    Write-Host "`n>>> Running PRESET: All College Images (Primary Pair + 3-Way Crisscross)" -ForegroundColor Green
    
    # 1. Primary pair
    $PrimaryImg1 = "C:\college\IMG_20240825_121630.jpg"
    $PrimaryImg2 = "C:\college\IMG_20240825_121629.jpg"
    Write-Host "`n[*] [STEP 1/2] Running Primary Image Pair: 121630 vs 121629..." -ForegroundColor Yellow
    
    & $PythonExe $RunnerScript `
        --img1 $PrimaryImg1 `
        --img2 $PrimaryImg2 `
        --detector $Detector `
        --max-kps $MaxKps `
        --high-res `
        --output-dir $OutputDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Primary pair benchmark failed with exit code $LASTEXITCODE"
    }

    # 2. Crisscross 3 images
    $CrisscrossImages = @(
        "C:\college\IMG_20240928_201329.jpg",
        "C:\college\IMG_20240928_201244.jpg",
        "C:\college\IMG_20240928_201211.jpg"
    )
    Write-Host "`n[*] [STEP 2/2] Running 3-Way Crisscross: 201329 vs 201244 vs 201211..." -ForegroundColor Yellow
    
    $crissArgs = @(
        $RunnerScript,
        "--crisscross",
        "--images"
    ) + $CrisscrossImages + @(
        "--detector", $Detector,
        "--max-kps", "$MaxKps",
        "--high-res",
        "--output-dir", $OutputDir
    )
    & $PythonExe @crissArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Crisscross benchmark failed with exit code $LASTEXITCODE"
    }

    Write-Host "`n==========================================================================" -ForegroundColor Cyan
    Write-Host "[+] All benchmarks and visualizations successfully completed!" -ForegroundColor Green
    Write-Host "    Check visual dashboard PNGs in: $(Resolve-Path $OutputDir)" -ForegroundColor Green
    Write-Host "==========================================================================`n" -ForegroundColor Cyan
    exit 0
}

# Standard invocation
$ArgsList = @($RunnerScript)

if ($Crisscross -or ($Images -and $Images.Count -gt 1)) {
    $ArgsList += "--crisscross"
    $ArgsList += "--images"
    if ($Images) {
        $flatImages = @()
        foreach ($item in $Images) {
            if ($item -match ",") {
                $flatImages += ($item -split "," | ForEach-Object { $_.Trim().Trim('"').Trim("'") } | Where-Object { $_ -ne "" })
            } else {
                $flatImages += $item.Trim().Trim('"').Trim("'")
            }
        }
        $ArgsList += $flatImages
    } else {
        $ArgsList += @($Img1, $Img2)
    }
} elseif ($Img1 -and $Img2) {
    $ArgsList += "--img1"
    $ArgsList += $Img1
    $ArgsList += "--img2"
    $ArgsList += $Img2
} else {
    Write-Host "[-] Usage error: Please provide -Img1 and -Img2, or -Crisscross -Images ..., or -AllCollege" -ForegroundColor Red
    Write-Host "    Example: .\run_quantum_cv.ps1 -Img1 'path1.jpg' -Img2 'path2.jpg'" -ForegroundColor Yellow
    exit 1
}

$ArgsList += "--detector"
$ArgsList += $Detector
$ArgsList += "--max-kps"
$ArgsList += "$MaxKps"

if ($HighRes) {
    $ArgsList += "--high-res"
}

$ArgsList += "--output-dir"
$ArgsList += $OutputDir

# Execute with PowerShell array splatting
& $PythonExe @ArgsList

if ($LASTEXITCODE -ne 0) {
    Write-Error "Quantum CV benchmark failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host "[+] Benchmark execution finished! Visual dashboards saved to $OutputDir" -ForegroundColor Green
Write-Host "==========================================================================`n" -ForegroundColor Cyan
