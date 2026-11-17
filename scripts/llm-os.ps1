param(
    [switch]$Gui,
    [switch]$Serial
)

$ErrorActionPreference = 'Stop'
if ($Gui -and $Serial) {
    throw 'Choose either -Gui or -Serial, not both.'
}

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$image = docker image inspect ai-os-lab:toolchain 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Toolchain image is missing. Run scripts\build.ps1 first.'
}

$prepare = "set -e; if [ ! -s models/stories260K.bin ] || [ ! -s models/tokenizer.bin ]; then make download-model; fi; make convert-model validate-model llm-iso"
docker run --rm --user 1000:1000 -e HOME=/tmp `
    --mount "type=bind,source=$root,target=/work" `
    --workdir /work ai-os-lab:toolchain sh -c $prepare
if ($LASTEXITCODE -ne 0) {
    throw 'LLM model preparation failed. Check internet access and retry.'
}

$hostQemu = Get-Command qemu-system-x86_64 -ErrorAction SilentlyContinue
if ($null -eq $hostQemu) {
    $installedQemu = 'C:\Program Files\qemu\qemu-system-x86_64.exe'
    if (Test-Path -LiteralPath $installedQemu) {
        $hostQemu = $installedQemu
    }
}
$qemuPath = if ($hostQemu -is [string]) { $hostQemu } else { $hostQemu.Source }
$useGui = $Gui -or (-not $Serial -and $null -ne $hostQemu)
if ($useGui) {
    if ($null -eq $hostQemu) {
        throw 'GUI mode requires qemu-system-x86_64 installed on Windows. Use .\scripts\llm-os.ps1 for serial mode, or install QEMU and retry with -Gui.'
    }
    Write-Host 'Starting native AI OS framebuffer GUI. Type prompts in the QEMU window.'
    & $qemuPath -m 256M -cdrom (Join-Path $root 'build\ai-os-llm.iso') `
        -display gtk -vga std -serial stdio `
        -device isa-debug-exit,iobase=0xf4,iosize=0x01 -no-reboot
    # isa-debug-exit encodes guest status 0 as host exit code 1.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 1) {
        throw 'Graphical LLM AI OS run failed.'
    }
} else {
    Write-Warning 'Host QEMU is not installed; using serial fallback. Install QEMU and run .\scripts\llm-os.ps1 -Gui for the framebuffer prompt window.'
    docker run --rm -it --user 1000:1000 -e HOME=/tmp `
        --mount "type=bind,source=$root,target=/work" `
        --workdir /work ai-os-lab:toolchain `
        qemu-system-x86_64 -m 256M -cdrom build/ai-os-llm.iso -display none -no-reboot -serial stdio
    if ($LASTEXITCODE -ne 0) {
        throw 'LLM AI OS run failed.'
    }
}
