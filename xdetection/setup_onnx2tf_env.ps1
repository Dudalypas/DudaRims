# Creates a fresh Python 3.12 environment for ONNX -> TFLite conversion.
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot '.venv-onnx2tf'
$ActivateScript = Join-Path $VenvPath 'Scripts\Activate.ps1'
$PythonExe = Join-Path $VenvPath 'Scripts\python.exe'
$PipExe = Join-Path $VenvPath 'Scripts\pip.exe'
$ConvertScript = Join-Path $ProjectRoot 'convert_onnx_to_tflite.py'

Write-Host '[convert] project root:' $ProjectRoot
Write-Host '[convert] create venv:' $VenvPath

if (Test-Path $VenvPath) {
    Write-Host '[convert] remove old venv'
    Remove-Item -Path $VenvPath -Recurse -Force
}

py -3.12 -m venv $VenvPath

if (-not (Test-Path $PythonExe)) {
    throw "[convert] failed to create venv Python executable: $PythonExe"
}

Write-Host '[convert] activate venv'
. $ActivateScript

Write-Host '[convert] upgrade pip'
& $PythonExe -m pip install --upgrade pip

Write-Host '[convert] remove conflicting packages'
& $PipExe uninstall -y tensorflow onnx2tf onnx onnx_graphsurgeon psutil onnxruntime onnxslim onnxsim sng4onnx tflite_support

Write-Host '[convert] install dependencies'
& $PipExe install `
    "tensorflow==2.19.1" `
    "onnx2tf[tensorflow]" `
    "onnx==1.18.0" `
    "onnx_graphsurgeon" `
    "psutil" `
    "onnxruntime" `
    "onnxslim" `
    "onnxsim" `
    "sng4onnx"

Write-Host '[convert] verify env'
& $PythonExe -c "import sys, tensorflow as tf, onnx, onnxruntime, onnxsim, onnxslim, sng4onnx, psutil; print('python', sys.version); print('tensorflow', tf.__version__); print('onnx', onnx.__version__)"

Write-Host ''
Write-Host '[convert] setup done'
Write-Host '[convert] run:'
Write-Host "& '$ProjectRoot\run_onnx_to_tflite.ps1'"
Write-Host '[convert] or run python:'
Write-Host "& '$PythonExe' '$ConvertScript'"
