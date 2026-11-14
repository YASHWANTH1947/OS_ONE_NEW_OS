# Probe Makefile: builds and boots the AVX2/FMA capability probe in QEMU
# using the project's Docker toolchain. Evidence gathering for the dispatch
# engine; not part of `make all`.
#
# Run from the repository root inside the toolchain container:
#   make -f probe/Makefile probe-run
#
# Or from the host:  docker run ... make -f probe/Makefile probe-run

# Repository root, derived from this Makefile's own location so the build
# works from any working directory (the container uses /work).
ROOT := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))/..
BUILD := $(ROOT)/build/probe
ISO_ROOT := $(BUILD)/iso-root
ISO := $(BUILD)/probe.iso
QEMU ?= qemu-system-x86_64
PYTHON ?= python3
CC := gcc
LD := ld

COMMON := -Wall -Wextra -Werror -std=c11 -ffreestanding -fno-builtin \
	-fno-stack-protector -fno-pic -fno-pie -fno-asynchronous-unwind-tables \
	-fno-unwind-tables -m64 -mcmodel=large -mno-red-zone -msse2 -mno-mmx \
	-O3 -Iprobe

CFLAGS_SCALAR := $(COMMON) -mno-avx
CFLAGS_AVX2   := $(COMMON) -mavx2
CFLAGS_FMA    := $(COMMON) -mavx2 -mfma

OBJECTS := $(BUILD)/probe-boot.o $(BUILD)/probe-main.o \
	$(BUILD)/probe-avx2.o $(BUILD)/probe-fma.o

.PHONY: probe probe-run probe-clean

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/probe-boot.o: $(ROOT)/boot/boot.S | $(BUILD)
	$(CC) -m64 -c -ffreestanding -fno-pic -fno-pie $< -o $@

$(BUILD)/probe-main.o: probe/probe.c | $(BUILD)
	$(CC) $(CFLAGS_SCALAR) -c $< -o $@

$(BUILD)/probe-avx2.o: probe/kernels_avx2.c | $(BUILD)
	$(CC) $(CFLAGS_AVX2) -c $< -o $@

$(BUILD)/probe-fma.o: probe/kernels_fma.c | $(BUILD)
	$(CC) $(CFLAGS_FMA) -c $< -o $@

$(BUILD)/probe.elf: $(OBJECTS) probe/linker.ld
	$(LD) -m elf_x86_64 -T probe/linker.ld -nostdlib -o $@ $(OBJECTS)

$(ISO): $(BUILD)/probe.elf probe/grub.cfg
	mkdir -p $(ISO_ROOT)/boot/grub
	cp $(BUILD)/probe.elf $(ISO_ROOT)/boot/probe.elf
	cp probe/grub.cfg $(ISO_ROOT)/boot/grub/grub.cfg
	grub-mkrescue -o $@ $(ISO_ROOT)

PROBE_CPU ?=
QEMU_ARGS :=

probe-run: $(ISO)
ifndef PROBE_CPU
	$(PYTHON) $(ROOT)/tools/run_qemu.py --qemu $(QEMU) --iso $(ISO) \
		--serial-file $(BUILD)/probe.log --timeout 120
else
	$(PYTHON) $(ROOT)/tools/run_qemu.py --qemu $(QEMU) --iso $(ISO) \
		--serial-file $(BUILD)/probe.log --timeout 120 --qemu-args "$(PROBE_CPU)"
endif

probe-clean:
	rm -rf $(BUILD)

# Allow `make -f probe/Makefile` from the repo root without further arguments.
.DEFAULT_GOAL := probe-run
