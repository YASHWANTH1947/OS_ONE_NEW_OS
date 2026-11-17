CC ?= gcc
LD ?= ld
PYTHON ?= python3
GRUB_MKRESCUE ?= grub-mkrescue
QEMU ?= qemu-system-x86_64

BUILD := build
ISO_ROOT := $(BUILD)/iso-root
ISO := $(BUILD)/ai-os-lab.iso
LLM_ISO_ROOT := $(BUILD)/iso-llm-root
LLM_ISO := $(BUILD)/ai-os-llm.iso
DASHBOARD_ISO_ROOT := $(BUILD)/iso-dashboard-root
DASHBOARD_ISO := $(BUILD)/ai-os-dashboard.iso
INCLUDES := -Iinclude
COMMON_CFLAGS := -Wall -Wextra -Werror $(INCLUDES)
KERNEL_CFLAGS := $(COMMON_CFLAGS) -std=c11 -ffreestanding -fno-builtin \
	-fno-stack-protector -fno-pic -fno-pie -fno-asynchronous-unwind-tables \
	-fno-unwind-tables -m64 -mcmodel=large -mno-red-zone -msse2 \
	-mno-mmx -O3
KERNEL_LDFLAGS := -m elf_x86_64 -T linker.ld -nostdlib

KERNEL_OBJECTS := $(BUILD)/boot.o $(BUILD)/isr.o $(BUILD)/kernel.o \
	$(BUILD)/serial.o $(BUILD)/memory.o $(BUILD)/interrupts.o \
	$(BUILD)/timer.o $(BUILD)/bench.o $(BUILD)/ai.o $(BUILD)/cpuid.o \
	$(BUILD)/text_ai.o $(BUILD)/ai_service.o $(BUILD)/llm_runtime.o \
	$(BUILD)/ui.o \
	$(BUILD)/kernels-scalar.o $(BUILD)/kernels-avx2.o $(BUILD)/kernels-fma.o

.PHONY: all iso llm-iso demo train download-model convert-model validate-model test run ai-os bench dashboard-mode benchmark dashboard serve clean \
	probe probe-max dispatch-csv

all: iso $(BUILD)/host-bench

iso: $(ISO)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/model.bin: | $(BUILD)
	@if test -s "$@"; then \
		echo "Using existing model: $@"; \
	else \
		$(PYTHON) tools/generate_demo_model.py "$@" || exit $$?; \
		touch $(BUILD)/demo-model; \
		rm -f $(BUILD)/training.json; \
	fi

$(BUILD)/samples.bin: | $(BUILD)
	@if test -s "$@"; then \
		echo "Using existing sample set: $@"; \
	else \
		$(PYTHON) tools/generate_demo_samples.py "$@" || exit $$?; \
		touch $(BUILD)/demo-samples; \
		rm -f $(BUILD)/training.json; \
	fi

$(BUILD)/text-model.bin: | $(BUILD)
	@if test -s "$@"; then \
		echo "Using existing text model: $@"; \
	else \
		$(PYTHON) tools/generate_demo_text_model.py "$@" || exit $$?; \
	fi

demo: | $(BUILD)
	rm -f $(BUILD)/training.json $(BUILD)/demo-model $(BUILD)/demo-samples
	$(PYTHON) tools/generate_demo_model.py $(BUILD)/model.bin
	$(PYTHON) tools/generate_demo_samples.py $(BUILD)/samples.bin
	touch $(BUILD)/demo-model $(BUILD)/demo-samples

train: | $(BUILD)
	rm -f $(BUILD)/demo-model $(BUILD)/demo-samples
	$(PYTHON) tools/train_mnist.py --output-dir $(BUILD)

download-model:
	$(PYTHON) tools/download_tinystories.py --output-dir models

MODEL_SOURCE ?= models/stories260K.bin
TOKENIZER_SOURCE ?= models/tokenizer.bin
LLM_MODEL ?= build/llm-model.bin
LLM_TOKENIZER ?= build/tokenizer.bin
LLM_MANIFEST ?= build/llm-manifest.json

convert-model:
	$(PYTHON) tools/package_tinystories.py \
		--checkpoint $(MODEL_SOURCE) \
		--tokenizer $(TOKENIZER_SOURCE) \
		--model-out $(LLM_MODEL) \
		--tokenizer-out $(LLM_TOKENIZER) \
		--manifest $(LLM_MANIFEST)

validate-model:
	$(PYTHON) tools/package_tinystories.py \
		--checkpoint $(MODEL_SOURCE) \
		--tokenizer $(TOKENIZER_SOURCE) \
		--model-out $(LLM_MODEL) \
		--tokenizer-out $(LLM_TOKENIZER) \
		--manifest $(LLM_MANIFEST) \
		--validate-only

$(BUILD)/boot.o: boot/boot.S | $(BUILD)
	$(CC) -m64 -c -ffreestanding -fno-pic -fno-pie $< -o $@

$(BUILD)/isr.o: boot/isr.S | $(BUILD)
	$(CC) -m64 -c -ffreestanding -fno-pic -fno-pie $< -o $@

$(BUILD)/%.o: src/%.c include/ai.h include/kernel.h include/bench.h \
	include/text_ai.h include/ai_service.h include/llm_runtime.h | $(BUILD)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

# Per-TU ISA flags (AEGIS-X System B): each kernel gets its own flags, so
# the engine/registry TU never contains a YMM instruction, and the SIMD TUs
# are only executed after the dispatcher proves eligibility + correctness.
$(BUILD)/kernels-scalar.o: src/kernels_scalar.c include/ai.h | $(BUILD)
	$(CC) $(KERNEL_CFLAGS) -mno-avx -c $< -o $@

$(BUILD)/kernels-avx2.o: src/kernels_avx2.c include/ai.h | $(BUILD)
	$(CC) $(KERNEL_CFLAGS) -mavx2 -c $< -o $@

$(BUILD)/kernels-fma.o: src/kernels_fma.c include/ai.h | $(BUILD)
	$(CC) $(KERNEL_CFLAGS) -mavx2 -mfma -c $< -o $@

$(BUILD)/kernel.elf: $(KERNEL_OBJECTS) linker.ld
	$(LD) $(KERNEL_LDFLAGS) -o $@ $(KERNEL_OBJECTS)

# ------------------------------------------------------------- host side
# Shared host objects with per-TU ISA flags, mirroring the unikernel build:
# the engine TU stays portable; the SIMD kernels carry their own flags.
HOST_CFLAGS := -std=c11 -O3 -Wall -Wextra -Werror $(INCLUDES)

HOST_ENGINE_OBJS := $(BUILD)/host-ai.o $(BUILD)/host-caps.o \
	$(BUILD)/host-kernels-scalar.o \
	$(BUILD)/host-kernels-avx2.o $(BUILD)/host-kernels-fma.o

$(BUILD)/host-ai.o: src/ai.c include/ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -c $< -o $@

$(BUILD)/host-caps.o: src/host_caps.c include/ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -c $< -o $@

$(BUILD)/host-kernels-scalar.o: src/kernels_scalar.c include/ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -c $< -o $@

$(BUILD)/host-kernels-avx2.o: src/kernels_avx2.c include/ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -mavx2 -c $< -o $@

$(BUILD)/host-kernels-fma.o: src/kernels_fma.c include/ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -mavx2 -mfma -c $< -o $@

# host_bench.c implements its own CPUID environment layer; src/cpuid.c is
# deliberately NOT linked here (it needs freestanding port I/O helpers).
$(BUILD)/host-bench: src/host_bench.c $(HOST_ENGINE_OBJS) include/ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) src/host_bench.c $(HOST_ENGINE_OBJS) -o $@

$(BUILD)/test-ai: tests/test_ai.c $(HOST_ENGINE_OBJS) include/ai.h | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(INCLUDES) \
		tests/test_ai.c $(HOST_ENGINE_OBJS) -o $@

# Kernel-equivalence tests: every eligible kernel must agree with the scalar
# reference (argmax exactly; logits within the shared gate) on real weights.
$(BUILD)/test-dispatch: tests/test_dispatch.c $(HOST_ENGINE_OBJS) include/ai.h | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(INCLUDES) \
		tests/test_dispatch.c $(HOST_ENGINE_OBJS) -o $@

$(BUILD)/host-ai-service.o: src/ai_service.c include/ai_service.h include/text_ai.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -c $< -o $@

$(BUILD)/host-llm-runtime.o: src/llm_runtime.c include/llm_runtime.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -c $< -o $@

$(BUILD)/llm_runtime.o: src/llm_runtime.c include/llm_runtime.h | $(BUILD)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD)/ui.o: src/ui.c include/ui.h include/kernel.h include/llm_runtime.h | $(BUILD)
	$(CC) $(KERNEL_CFLAGS) -c $< -o $@

$(BUILD)/test-ai-service: tests/test_ai_service.c $(BUILD)/host-ai-service.o \
	include/ai_service.h include/text_ai.h | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(INCLUDES) \
		tests/test_ai_service.c $(BUILD)/host-ai-service.o -o $@

$(BUILD)/test-llm-runtime: tests/test_llm_runtime.c $(BUILD)/host-llm-runtime.o \
	include/llm_runtime.h | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(INCLUDES) \
		tests/test_llm_runtime.c $(BUILD)/host-llm-runtime.o -o $@

$(BUILD)/host-ui.o: src/ui.c include/ui.h include/kernel.h include/llm_runtime.h | $(BUILD)
	$(CC) $(HOST_CFLAGS) -c $< -o $@

$(BUILD)/test-ui: tests/test_ui.c $(BUILD)/host-ui.o $(BUILD)/host-llm-runtime.o \
	include/ui.h include/kernel.h | $(BUILD)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(INCLUDES) \
		tests/test_ui.c $(BUILD)/host-ui.o $(BUILD)/host-llm-runtime.o -o $@

test: $(BUILD)/test-ai $(BUILD)/test-dispatch $(BUILD)/test-ai-service \
	$(BUILD)/test-llm-runtime $(BUILD)/test-ui \
	$(BUILD)/model.bin $(BUILD)/samples.bin $(BUILD)/text-model.bin
	$(BUILD)/test-ai $(BUILD)/model.bin $(BUILD)/samples.bin
	$(BUILD)/test-dispatch $(BUILD)/model.bin $(BUILD)/samples.bin
	$(BUILD)/test-ai-service $(BUILD)/text-model.bin
	$(BUILD)/test-llm-runtime
	$(BUILD)/test-ui
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

$(BUILD)/host.json: $(BUILD)/host-bench $(BUILD)/model.bin $(BUILD)/samples.bin
	$(BUILD)/host-bench $(BUILD)/model.bin $(BUILD)/samples.bin \
		--passes 10000 --json $@ > $(BUILD)/host.log 2>&1 \
		|| { cat $(BUILD)/host.log; exit 1; }

# AEGIS-X factorial cells (E1 scalar, E2 avx2/fma on Linux): append one CSV
# row per forced kernel, reproducing the methodology of the unikernel cells.
dispatch-csv: $(BUILD)/host-bench $(BUILD)/model.bin $(BUILD)/samples.bin
	rm -f $(BUILD)/dispatch.csv
	$(BUILD)/host-bench $(BUILD)/model.bin $(BUILD)/samples.bin \
		--passes 10000 --kernel scalar --csv $(BUILD)/dispatch.csv \
		> $(BUILD)/host-scalar.log 2>&1 || { cat $(BUILD)/host-scalar.log; exit 1; }
	$(BUILD)/host-bench $(BUILD)/model.bin $(BUILD)/samples.bin \
		--passes 10000 --kernel avx2 --csv $(BUILD)/dispatch.csv \
		> $(BUILD)/host-avx2.log 2>&1 \
		|| { echo "avx2 unavailable on this host" && cat $(BUILD)/host-avx2.log && exit 0; }
	$(BUILD)/host-bench $(BUILD)/model.bin $(BUILD)/samples.bin \
		--passes 10000 --kernel fma --csv $(BUILD)/dispatch.csv \
		> $(BUILD)/host-fma.log 2>&1 \
		|| { echo "fma unavailable on this host" && cat $(BUILD)/host-fma.log && exit 0; }
	cat $(BUILD)/dispatch.csv

$(ISO): $(BUILD)/kernel.elf $(BUILD)/model.bin $(BUILD)/samples.bin \
	$(BUILD)/text-model.bin grub/grub.cfg
	mkdir -p $(ISO_ROOT)/boot/grub
	cp $(BUILD)/kernel.elf $(ISO_ROOT)/boot/kernel.elf
	cp $(BUILD)/model.bin $(ISO_ROOT)/boot/model.bin
	cp $(BUILD)/samples.bin $(ISO_ROOT)/boot/samples.bin
	cp $(BUILD)/text-model.bin $(ISO_ROOT)/boot/text-model.bin
	cp grub/grub.cfg $(ISO_ROOT)/boot/grub/grub.cfg
	$(GRUB_MKRESCUE) -o $@ $(ISO_ROOT)

llm-iso: $(LLM_ISO)

$(LLM_ISO): $(BUILD)/kernel.elf $(BUILD)/model.bin $(BUILD)/samples.bin \
	$(BUILD)/text-model.bin $(LLM_MODEL) $(LLM_TOKENIZER) grub/grub-llm.cfg
	mkdir -p $(LLM_ISO_ROOT)/boot/grub/fonts
	cp $(BUILD)/kernel.elf $(LLM_ISO_ROOT)/boot/kernel.elf
	cp $(BUILD)/model.bin $(LLM_ISO_ROOT)/boot/model.bin
	cp $(BUILD)/samples.bin $(LLM_ISO_ROOT)/boot/samples.bin
	cp $(BUILD)/text-model.bin $(LLM_ISO_ROOT)/boot/text-model.bin
	cp $(LLM_MODEL) $(LLM_ISO_ROOT)/boot/llm-model.bin
	cp $(LLM_TOKENIZER) $(LLM_ISO_ROOT)/boot/tokenizer.bin
	cp grub/grub-llm.cfg $(LLM_ISO_ROOT)/boot/grub/grub.cfg
	cp /usr/share/grub/unicode.pf2 $(LLM_ISO_ROOT)/boot/grub/fonts/unicode.pf2
	$(GRUB_MKRESCUE) -o $@ $(LLM_ISO_ROOT)

$(BUILD)/dashboard.flag: | $(BUILD)
	printf 'dashboard\n' > $@

$(DASHBOARD_ISO): $(BUILD)/kernel.elf $(BUILD)/model.bin $(BUILD)/samples.bin \
	$(BUILD)/text-model.bin $(BUILD)/dashboard.flag grub/grub-dashboard.cfg
	mkdir -p $(DASHBOARD_ISO_ROOT)/boot/grub
	cp $(BUILD)/kernel.elf $(DASHBOARD_ISO_ROOT)/boot/kernel.elf
	cp $(BUILD)/model.bin $(DASHBOARD_ISO_ROOT)/boot/model.bin
	cp $(BUILD)/samples.bin $(DASHBOARD_ISO_ROOT)/boot/samples.bin
	cp $(BUILD)/text-model.bin $(DASHBOARD_ISO_ROOT)/boot/text-model.bin
	cp $(BUILD)/dashboard.flag $(DASHBOARD_ISO_ROOT)/boot/dashboard.flag
	cp grub/grub-dashboard.cfg $(DASHBOARD_ISO_ROOT)/boot/grub/grub.cfg
	$(GRUB_MKRESCUE) -o $@ $(DASHBOARD_ISO_ROOT)

run: $(ISO)
	$(PYTHON) tools/run_qemu.py --qemu "$(QEMU)" --iso $(ISO) --serial stdio

# AI OS mode: boot the ring-0 inference runtime and leave its serial shell
# available for interactive predictions and status inspection.
ai-os: run

# Full pipeline: boot the unikernel in QEMU, capture the serial transcript,
# score the same samples natively, then compile both into the dashboard data.
bench: $(DASHBOARD_ISO) $(BUILD)/host.json
	$(PYTHON) tools/run_qemu.py --qemu "$(QEMU)" --iso $(DASHBOARD_ISO) \
		--serial-file $(BUILD)/unikernel.log --expect-benchmark
	$(PYTHON) tools/make_report.py \
		--unikernel-log $(BUILD)/unikernel.log \
		--host-json $(BUILD)/host.json \
		--host-log $(BUILD)/host.log \
		$(if $(wildcard $(BUILD)/training.json),--training $(BUILD)/training.json,) \
		$(if $(wildcard $(BUILD)/demo-model),--demo-model,) \
		$(if $(wildcard $(BUILD)/demo-samples),--demo-samples,) \
		--kernel $(BUILD)/kernel.elf \
		--model $(BUILD)/model.bin \
		--samples $(BUILD)/samples.bin \
		--iso $(DASHBOARD_ISO) \
		--out-dir dashboard/data

benchmark: bench

# Dashboard mode: capture a complete AI OS run, compare it with the matched
# host baseline, and regenerate the static browser report.
dashboard-mode: bench
dashboard: dashboard-mode

# Capability probes from the dispatch-engine evidence gathering.
probe:
	$(MAKE) -f probe/Makefile probe-run
probe-max:
	$(MAKE) -f probe/Makefile probe-run PROBE_CPU='-cpu max'

serve: dashboard
	$(PYTHON) -m http.server 8000 --directory dashboard

clean:
	rm -rf $(BUILD)
