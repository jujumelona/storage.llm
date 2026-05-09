$ErrorActionPreference = "Stop"

Write-Host "[storageLLM] C++ engine path: configure/build/autotune/select automatically"
Write-Host "[storageLLM] Python is helper-only for pip TVM/codegen, not runtime orchestration."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target storagellm_host_autotune --parallel

$exeCandidates = @(
  ".\build\Release\storagellm_host_autotune.exe",
  ".\build\storagellm_host_autotune.exe",
  ".\build\Debug\storagellm_host_autotune.exe",
  "./build/storagellm_host_autotune"
)
$exe = $null
foreach ($c in $exeCandidates) {
  if (Test-Path $c) { $exe = $c; break }
}
if (-not $exe) { throw "storagellm_host_autotune executable was not built" }

& $exe
if ($LASTEXITCODE -ne 0) {
  Write-Host "[storageLLM] autotune did not select a measured fast backend; continuing fail-closed. See build/auto_backend_report.json"
}

cmake --build build --config Release --parallel

if (Test-Path .\build\selected_backend.env) {
  Get-Content .\build\selected_backend.env | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
      [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
  }
}
Write-Host "[storageLLM] done. Report: build/auto_backend_report.json"
