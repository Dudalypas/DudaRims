# Activates the conversion environment and runs the ONNX -> TFLite converter.
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot '.venv-onnx2tf'
$ActivateScript = Join-Path $VenvPath 'Scripts\Activate.ps1'
$PythonExe = Join-Path $VenvPath 'Scripts\python.exe'
$ConvertScript = Join-Path $ProjectRoot 'convert_onnx_to_tflite.py'

if (-not (Test-Path $ActivateScript)) {
    Write-Host '[convert] missing venv script:' $ActivateScript
    Write-Host '[convert] run setup first:'
    Write-Host "& '$ProjectRoot\setup_onnx2tf_env.ps1'"
    exit 1
}

if (-not (Test-Path $ConvertScript)) {
    Write-Host '[convert] missing converter:' $ConvertScript
    exit 1
}

Write-Host '[convert] activate venv'
. $ActivateScript

Write-Host '[convert] run converter'

$stdoutFile = Join-Path $env:TEMP 'onnx2tf_convert_stdout.log'
$stderrFile = Join-Path $env:TEMP 'onnx2tf_convert_stderr.log'

if (Test-Path $stdoutFile) { Remove-Item $stdoutFile -Force }
if (Test-Path $stderrFile) { Remove-Item $stderrFile -Force }

$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList @($ConvertScript) `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput $stdoutFile `
    -RedirectStandardError $stderrFile

Write-Host '--- STDOUT ---'
if (Test-Path $stdoutFile) {
    Get-Content $stdoutFile
}

Write-Host '--- STDERR ---'
if (Test-Path $stderrFile) {
    Get-Content $stderrFile
}

if ($proc.ExitCode -eq 0) {
    Write-Host '[convert] done'
    exit 0
}

Write-Host "[convert] failed: exit $($proc.ExitCode)"
exit $proc.ExitCode