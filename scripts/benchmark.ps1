$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$image = docker image inspect ai-os-lab:toolchain 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Toolchain image is missing. Run scripts\build.ps1 first.'
}

docker run --rm --user 1000:1000 -e HOME=/tmp `
    --mount "type=bind,source=$root,target=/work" `
    --workdir /work ai-os-lab:toolchain make bench
if ($LASTEXITCODE -ne 0) {
    throw 'Benchmark run failed; see build/unikernel.log and build/bench-run.log.'
}

Write-Host ''
Write-Host 'Measured report: dashboard/index.html (data in dashboard/data/results.json)'
