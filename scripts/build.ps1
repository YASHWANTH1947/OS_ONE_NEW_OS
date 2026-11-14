$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

docker build -f (Join-Path $root 'Dockerfile.toolchain') `
    -t ai-os-lab:toolchain $root
if ($LASTEXITCODE -ne 0) {
    throw 'Docker toolchain image build failed.'
}

# Run as a non-root uid so the generated files stay writable from the host.
docker run --rm --user 1000:1000 -e HOME=/tmp `
    --mount "type=bind,source=$root,target=/work" `
    --workdir /work ai-os-lab:toolchain make all
if ($LASTEXITCODE -ne 0) {
    throw 'AI OS lab build failed.'
}
