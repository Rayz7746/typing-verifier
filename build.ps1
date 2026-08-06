[CmdletBinding()]
param(
    [ValidatePattern('^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$')]
    [string]$Version = '0.1.0',

    [ValidateRange(3, 5)]
    [int]$SmokeTestSeconds = 5
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot 'build'
$DistDir = Join-Path $ProjectRoot 'dist'
$SpecPath = Join-Path $ProjectRoot 'TypingVerifier.spec'
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$BundleDir = Join-Path $DistDir 'TypingVerifier'
$ExePath = Join-Path $BundleDir 'TypingVerifier.exe'
$InternalDir = Join-Path $BundleDir '_internal'
$ModelPath = Join-Path $InternalDir 'models\hand_landmarker.task'
$ArchiveName = "TypingVerifier-v$Version-windows-x64.zip"
$ArchivePath = Join-Path $DistDir $ArchiveName
$ChecksumPath = "$ArchivePath.sha256"
$SmokeLocalAppData = Join-Path $BuildDir 'smoke-localappdata'
$SmokeLogPath = Join-Path $SmokeLocalAppData 'TypingVerifier\logs\typing-verifier.log'

function Remove-BuildDirectory {
    param([Parameter(Mandatory)][string]$Path)

    $resolvedRoot = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $resolvedTarget = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not $resolvedTarget.StartsWith("$resolvedRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the project: $resolvedTarget"
    }
    if (Test-Path -LiteralPath $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Project Python was not found at $PythonPath. Create .venv and install requirements first."
}
if (-not (Test-Path -LiteralPath $SpecPath -PathType Leaf)) {
    throw "PyInstaller spec file was not found at $SpecPath."
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'models\hand_landmarker.task') -PathType Leaf)) {
    throw 'models\hand_landmarker.task is missing. Run: python .\fetch_model.py'
}

& $PythonPath -c 'import PyInstaller' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is not installed. Run: .\.venv\Scripts\python.exe -m pip install -r .\requirements-build.txt'
}

Remove-BuildDirectory -Path $BuildDir
Remove-BuildDirectory -Path $DistDir

Push-Location $ProjectRoot
try {
    & $PythonPath -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

foreach ($requiredPath in @($ExePath, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required packaged file is missing: $requiredPath"
    }
}

New-Item -ItemType Directory -Path $SmokeLocalAppData -Force | Out-Null
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $ExePath
$startInfo.WorkingDirectory = $BundleDir
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$startInfo.EnvironmentVariables['LOCALAPPDATA'] = $SmokeLocalAppData

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
try {
    if (-not $process.Start()) {
        throw 'Smoke test process could not be started.'
    }

    $expectedWindowTitle = 'Real-Time Touch-Typing Verifier'
    $deadline = [DateTime]::UtcNow.AddSeconds($SmokeTestSeconds)
    $mainWindowReady = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) {
            throw "Smoke test process exited early with code $($process.ExitCode)."
        }
        $process.Refresh()
        if ($process.MainWindowTitle -eq $expectedWindowTitle) {
            $mainWindowReady = $true
            break
        }
        Start-Sleep -Milliseconds 200
    }

    if (-not $mainWindowReady) {
        $observedTitle = $process.MainWindowTitle
        throw "Smoke test did not reach the main window. Observed title: '$observedTitle'."
    }
}
finally {
    if (-not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
    $process.Dispose()
}

if (-not (Test-Path -LiteralPath $SmokeLogPath -PathType Leaf)) {
    throw "Smoke test log was not created: $SmokeLogPath"
}
$smokeLog = Get-Content -LiteralPath $SmokeLogPath -Raw
if ($smokeLog -match '\[CRITICAL\]|Traceback \(most recent call last\)|Uncaught exception') {
    throw "Smoke test log contains a fatal exception. Review: $SmokeLogPath"
}

Compress-Archive -LiteralPath $BundleDir -DestinationPath $ArchivePath -CompressionLevel Optimal
$checksum = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$checksum  $ArchiveName" | Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Write-Host "Build completed successfully:"
Write-Host "  Bundle:   $BundleDir"
Write-Host "  Archive:  $ArchivePath"
Write-Host "  Checksum: $ChecksumPath"
