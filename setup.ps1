<#
.SYNOPSIS
    Setup script for RealtimeGameTranslation.
    Checks Python, virtual environment, dependencies, and system requirements (like VC++ Redist).
#>

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "RealtimeGameTranslation - Setup Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python installation
Write-Host "1. Checking Python installation..." -ForegroundColor Yellow
$pythonExe = ""
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
} elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $pythonExe = "python3"
}

if (-not $pythonExe) {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.9 or newer from https://www.python.org/downloads/"
    exit 1
}

$pyVerStr = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "Found Python version: $pyVerStr" -ForegroundColor Green

# 2. Virtual Environment
Write-Host "`n2. Checking Virtual Environment..." -ForegroundColor Yellow
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating Virtual Environment in .venv..."
    & $pythonExe -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create Virtual Environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual Environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual Environment already exists." -ForegroundColor Green
}

# 3. Install Requirements
Write-Host "`n3. Installing/Verifying Dependencies..." -ForegroundColor Yellow
$pipExe = Join-Path $venvPath "Scripts\pip.exe"
if (-not (Test-Path $pipExe)) {
    Write-Host "ERROR: pip not found in virtual environment." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..."
& $pipExe install --upgrade pip -q

$requirementsFile = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing packages from requirements.txt..."
    # Esegue l'installazione
    $process = Start-Process -FilePath $pipExe -ArgumentList "install -r `"$requirementsFile`"" -Wait -NoNewWindow -PassThru
    if ($process.ExitCode -ne 0) {
        Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements.txt not found!" -ForegroundColor Yellow
}

# 4. Check for VC++ Redistributable (Required for ONNXRuntime / OCR)
Write-Host "`n4. Checking System Requirements (VC++ Redistributable)..." -ForegroundColor Yellow
$vcRedistInstalled = $false
$registryPaths = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)

foreach ($path in $registryPaths) {
    if (Test-Path $path) {
        $props = Get-ItemProperty -Path $path -ErrorAction SilentlyContinue
        if ($null -ne $props.Installed -and $props.Installed -eq 1) {
            $vcRedistInstalled = $true
            break
        }
    }
}

if ($vcRedistInstalled) {
    Write-Host "Visual C++ Redistributable x64 is installed." -ForegroundColor Green
} else {
    Write-Host "WARNING: Visual C++ Redistributable x64 is MISSING!" -ForegroundColor Red
    Write-Host "This is strictly required for ONNXRuntime (used by the OCR engine)." -ForegroundColor Yellow
    $choice = Read-Host "Do you want to download and install it automatically now? (Y/N)"
    if ($choice -match "^[yY]") {
        $downloadUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        $installerPath = Join-Path $env:TEMP "vc_redist.x64.exe"
        Write-Host "Downloading VC++ Redistributable..."
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath
        Write-Host "Installing VC++ Redistributable (this may prompt for Administrator permissions)..."
        $process = Start-Process -FilePath $installerPath -ArgumentList "/install /quiet /norestart" -Wait -PassThru
        if ($process.ExitCode -eq 0 -or $process.ExitCode -eq 3010) {
            Write-Host "Installation successful." -ForegroundColor Green
            if ($process.ExitCode -eq 3010) {
                Write-Host "NOTE: A system reboot might be required to complete VC++ Redistributable installation." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Installation failed with exit code $($process.ExitCode). Please install it manually from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist" -ForegroundColor Red
        }
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "Skipping automatic installation. The program WILL CRASH during OCR if you don't install it." -ForegroundColor Yellow
        Write-Host "Download it from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist" -ForegroundColor Yellow
    }
}

# 5. Fix for models folder being empty / gitignore issues
Write-Host "`n5. Verifying internal directories..." -ForegroundColor Yellow
$modelsDir = Join-Path $PSScriptRoot "models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
    Write-Host "Created 'models' directory." -ForegroundColor Green
}
$gitkeepPath = Join-Path $modelsDir ".gitkeep"
if (-not (Test-Path $gitkeepPath)) {
    Set-Content -Path $gitkeepPath -Value "Keep directory for model downloads"
    Write-Host "Created .gitkeep in models directory to track it in git." -ForegroundColor Green
}

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "Setup Completed successfully!" -ForegroundColor Green
Write-Host "You can now run the application using the virtual environment."
Write-Host "To start the app, you can use: .venv\Scripts\python.exe main.py"
Write-Host "=========================================`n" -ForegroundColor Cyan
