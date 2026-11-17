# Bare-Metal AI Unikernel OS Plan

## Mission

Build a bare-metal x86_64 AI operating system that runs a local language model
directly above the kernel in ring 0. Model loading, tokenization, tensor
inference, keyboard input, and display rendering must execute inside the
freestanding kernel.Act as a Principal Systems & Kernel Engineer. We are refactoring and building our bare-metal x86_64 AI Unikernel OS.

### Strict Architectural Constraints (DO NOT VIOLATE)

1. **100% Bare-Metal Execution (Ring 0):** Everything—including model loading, tokenization, tensor inference, keyboard input, and display rendering—must execute directly inside our freestanding x86_64 kernel.
2. **NO Host-Side Shortcuts:** Do NOT propose building a web GUI, Electron app, or host-side Python frontend connected via serial. All UI/display rendering must happen natively inside the kernel (e.g., via Multiboot2 framebuffer / VESA linear framebuffer pixel blitting and PS/2 keyboard interrupts).
3. **No Custom LLM Training:** We are NOT training an LLM from scratch. We will download an existing, pre-trained open-source micro-model (such as Karpathy's `stories15M.bin` or `stories260K.bin` from `llama2.c` / TinyStories) and write a minimal Python conversion script to pack its weights and tokenizer into flat, little-endian binary files (`llm-model.bin`, `tokenizer.bin`) suitable for GRUB Multiboot2 module loading.
4. **Freestanding C Rules:** No `libc`, no `malloc`/`free` during inference, no filesystem assumptions. Use a static, pre-allocated memory arena (`kmalloc` bump allocator or fixed arena) and explicit bounded buffers.

### Current Foundation

- x86_64 kernel boots via GRUB (Multiboot2) in Ring 0.
- Adaptive scalar/AVX2/FMA math routines and a serial shell are working.
- A basic deterministic demo model currently runs via an `ask` command.

### Immediate Execution Plan (Step-by-Step)

Let's execute this cleanly one phase at a time. Do not dump all code at once—confirm each step before moving to the next:

- **Step 1 (Kernel AI Service API):** Refactor our existing demo model into a clean header/implementation (`ai_service.h` / `ai_service.c`) with explicit error codes, bounded buffers, and functions: `ai_init_arena()`, `ai_load_model()`, `ai_generate_stream()`, and `ai_get_metrics()` (tracking CPU cycles via `rdtsc` and memory footprint).
- **Step 2 (Pre-trained Model Packaging):** Provide a clean Python script to fetch `stories260K` (or `stories15M`), validate its header, and pack the weights + tokenizer into binary blobs that GRUB loads as Multiboot2 modules into physical RAM.
- **Step 3 (Pure C Transformer Runtime):** Implement the freestanding Llama2-style forward pass in C (RMSNorm, RoPE, Multi-Head Attention, SwiGLU, MatMul using our existing AVX2/scalar backend) operating strictly inside our static memory arena.
- **Step 4 (Native Framebuffer UI & PS/2 Input):** Request a linear framebuffer via the Multiboot2 header tag, implement a lightweight 8x16 bitmap font renderer in Ring 0, and wire up PS/2 keyboard IRQ1 interrupts so the user can type prompts directly on the bare-metal screen and watch tokens stream live.

Start right now with **Step 1**: inspect our current codebase structure and propose the exact `ai_service.h` API contract and memory arena layout.

The final boot flow is:

```text
GRUB
  -> x86_64 kernel enters long mode
  -> kernel initializes memory, interrupts, display, and input
  -> kernel loads local model and tokenizer modules
  -> kernel initializes the AI runtime
  -> native framebuffer UI accepts a prompt
  -> the model generates tokens directly in ring 0
  -> generated tokens are rendered on the framebuffer
```

## Non-negotiable architectural constraints

1. **100% bare-metal execution**
   - All AI execution stays inside the freestanding x86_64 kernel.
   - Model loading, tokenization, inference, keyboard input, and display
     rendering run in ring 0.
   - There is no user/kernel process boundary, scheduler dependency, or host
     inference service.

2. **No host-side UI or inference shortcuts**
   - Do not build a web GUI, Electron application, or host-side Python
     frontend.
   - Do not run a second copy of the model on the host.
   - The final UI is rendered natively by the kernel using a Multiboot2
     framebuffer or VESA linear framebuffer.
   - Input is provided by native PS/2 keyboard handling and IRQ1.

3. **Use an existing pre-trained micro-model**
   - Do not train an LLM from scratch.
   - Download an existing open-source model such as Karpathy's
     `stories260K.bin` or `stories15M.bin` from `llama2.c` / TinyStories.
   - Convert the model and tokenizer into flat, little-endian binary files:
     - `llm-model.bin`
     - `tokenizer.bin`
   - Load both files as GRUB Multiboot2 modules into physical RAM.

4. **Freestanding C only**
   - No libc, filesystem, processes, or host runtime assumptions.
   - No `malloc` or `free` during inference.
   - Use a static, pre-allocated memory arena, backed initially by the
     existing bump allocator and later by reserved physical frames.
   - Use explicit sizes, bounded buffers, fixed layouts, and explicit errors.

5. **Incremental execution**
   - Execute one phase at a time.
   - Do not dump all implementation code at once.
   - Inspect the current code, implement the current phase, validate it, and
     confirm before moving to the next phase.

## Current foundation

Already present in the repository:

- GRUB/Multiboot2 x86_64 boot path.
- Long-mode kernel running in ring 0.
- Identity paging and physical frame allocator.
- 64 KiB kernel bump heap.
- Serial console and interactive kernel shell.
- `RDTSC` timing and PIT calibration.
- Scalar, AVX2, and AVX2+FMA inference kernels.
- Measured SIMD dispatch with correctness gates.
- Existing 784-128-10 neural-network demo model.
- Existing deterministic character-model demo through `ask TEXT`.
- Docker-based build and QEMU execution workflow.

Current limitations:

- `text_ai.c` is a deterministic demo model, not a useful LLM.
- The current text generation API is not a reusable AI service.
- The existing 64 KiB heap cannot hold a copied `stories15M` model.
- Current UI is serial-only.
- There is no native framebuffer renderer or PS/2 keyboard interface.
- The TinyStories model is not yet converted into a kernel-friendly package.

## Target architecture

```text
+--------------------------------------------------------------+
| Native kernel UI                                              |
| framebuffer renderer + bitmap font + PS/2 keyboard IRQ1       |
+--------------------------------------------------------------+
| Kernel AI service                                             |
| bounded prompt API + token streaming + metrics                |
+--------------------------------------------------------------+
| Tokenizer                                                     |
| tokenizer.bin, token IDs, bounded context                     |
+--------------------------------------------------------------+
| Transformer runtime                                           |
| RMSNorm + RoPE + attention + SwiGLU + MatMul                  |
+--------------------------------------------------------------+
| Math backend                                                  |
| scalar / AVX2 / FMA kernels with measured dispatch             |
+--------------------------------------------------------------+
| Static AI arena                                               |
| activations + temporary buffers + KV cache                    |
+--------------------------------------------------------------+
| GRUB Multiboot2 modules                                      |
| llm-model.bin + tokenizer.bin                                |
+--------------------------------------------------------------+
| x86_64 freestanding kernel                                    |
+--------------------------------------------------------------+
```

Model weights and tokenizer data should be used zero-copy from their
Multiboot2 module addresses. They must not be copied into the small general
kernel heap.

## Phased implementation plan

### Step 1 — Kernel AI service API

Refactor the existing demo model into:

- `include/ai_service.h`
- `src/ai_service.c`

Required operations:

- `ai_init_arena()`
- `ai_load_model()`
- `ai_generate_stream()`
- `ai_get_metrics()`

Requirements:

- Explicit status/error enum.
- Caller-owned `ai_service_t` state.
- Caller-owned static arena.
- Zero-copy model attachment.
- Bounded prompt, token, and output buffers.
- No allocation during generation.
- Token callback for streaming output.
- `RDTSC` cycle metrics.
- Arena capacity, arena usage, model size, token count, and generation
  statistics.

Proposed initial service API:

```c
typedef enum {
    AI_SERVICE_OK = 0,
    AI_SERVICE_ERR_ARGUMENT,
    AI_SERVICE_ERR_NOT_INITIALIZED,
    AI_SERVICE_ERR_ALREADY_LOADED,
    AI_SERVICE_ERR_MODEL_NULL,
    AI_SERVICE_ERR_MODEL_SIZE,
    AI_SERVICE_ERR_MODEL_FORMAT,
    AI_SERVICE_ERR_MODEL_VERSION,
    AI_SERVICE_ERR_ARENA_NULL,
    AI_SERVICE_ERR_ARENA_SIZE,
    AI_SERVICE_ERR_ARENA_EXHAUSTED,
    AI_SERVICE_ERR_PROMPT_TOO_LONG,
    AI_SERVICE_ERR_OUTPUT_TOO_SMALL,
    AI_SERVICE_ERR_GENERATION_LIMIT,
    AI_SERVICE_ERR_CALLBACK
} ai_service_status_t;

typedef int (*ai_token_callback_t)(const char *text,
                                   size_t bytes,
                                   uint32_t token_index,
                                   void *context);

ai_service_status_t ai_init_arena(ai_service_t *service,
                                  void *arena,
                                  size_t arena_bytes);

ai_service_status_t ai_load_model(ai_service_t *service,
                                  const void *model,
                                  size_t model_bytes);

ai_service_status_t ai_generate_stream(ai_service_t *service,
                                        const char *prompt,
                                        size_t prompt_bytes,
                                        const ai_generate_config_t *config,
                                        ai_token_callback_t callback,
                                        void *callback_context);

ai_service_status_t ai_get_metrics(const ai_service_t *service,
                                    ai_metrics_t *metrics);
```

Initial demo arena target:

```text
16 KiB, 64-byte aligned
```

Initial arena layout:

```text
service state
prompt buffer
token/index scratch
generated output buffer
current hidden state
next hidden state
aligned temporary scratch
```

Step 1 validation:

- Invalid service and arena arguments fail explicitly.
- Invalid model headers fail explicitly.
- Model data is not copied.
- Prompt and output limits are enforced.
- Callback receives generated output.
- Metrics update after generation.
- Existing `ask TEXT` behavior remains available through the service layer.

### Step 2 — Pre-trained model packaging

Add a minimal Python conversion tool that:

1. Downloads `stories260K` first; support `stories15M` as an optional larger
   target.
2. Validates the source model header, dimensions, vocabulary, and file size.
3. Validates tokenizer metadata and vocabulary.
4. Emits flat little-endian files:
   - `build/llm-model.bin`
   - `build/tokenizer.bin`
5. Emits a manifest containing dimensions, checksums, and byte sizes.
6. Never trains a model.

Add Makefile targets:

```text
make download-model
make convert-model
make validate-model
```

The model format must be versioned and include enough metadata for the kernel
to reject incompatible files before inference.

### Step 3 — Pure C Transformer runtime

Implement a freestanding C inference runtime operating only on the static AI
arena and zero-copy model module:

- Token embedding lookup.
- RMSNorm.
- Rotary positional embeddings.
- Multi-head attention.
- Key/value cache.
- SwiGLU feed-forward layers.
- MatMul using scalar and existing SIMD backends.
- Final normalization and logits.
- Greedy decoding first.
- Sampling only after deterministic generation is working.

No libc, filesystem, dynamic allocation, or host calls are permitted.

Initial runtime target:

- `stories260K`.
- Fixed context length, initially 128 tokens.
- Fixed maximum output length, initially 128 tokens.
- Float32 correctness implementation first.
- Int8/int4 quantization after functional validation.

Expected arena regions:

```text
service/context metadata
tokenizer scratch
prompt token IDs
generated token IDs
current activation
next activation
attention temporary buffers
KV cache
SIMD-aligned scratch
```

Approximate arena targets:

| Model              | Initial AI arena target |
| ------------------ | ----------------------: |
| Current demo model |                  16 KiB |
| `stories260K`      |                   2 MiB |
| `stories15M`       |               64–96 MiB |

The memory subsystem must be extended before the larger models are used.

### Step 4 — Native framebuffer UI and PS/2 input

Implement all UI inside the kernel:

1. Request a linear framebuffer through the Multiboot2 header/tag.
2. Parse framebuffer information during boot.
3. Implement a bounded pixel-blit primitive.
4. Add an 8x16 bitmap font.
5. Implement text cursor, line wrapping, scrolling, and prompt editing.
6. Initialize the PS/2 controller.
7. Install IRQ1 keyboard handling.
8. Decode scan codes into bounded ASCII/keyboard events.
9. Render generated tokens as they arrive from `ai_generate_stream()`.

The UI must remain a kernel component and must not depend on a host GUI.

### Step 5 — Boot/module integration

Extend the GRUB configuration and kernel module discovery for:

```text
kernel.elf
llm-model.bin
tokenizer.bin
```

Add:

- Module name validation.
- Module bounds validation.
- Model and tokenizer checksum validation.
- Memory footprint reporting.
- Explicit boot errors for missing or incompatible modules.
- Dedicated AI arena reservation.

### Step 6 — Measurement and optimization

Measure inside the kernel:

- Boot-to-AI-ready cycles.
- Model validation cycles.
- Prompt tokenization cycles.
- Prefill latency.
- Per-token generation cycles.
- Total generation cycles.
- Tokens per second.
- Peak arena usage.
- KV-cache usage.
- Framebuffer rendering cost.

Optimization order:

1. Correctness.
2. Bounds and failure handling.
3. Arena layout and memory locality.
4. Quantization.
5. Scalar baseline.
6. AVX2/FMA kernels and measured dispatch.
7. Streaming and UI latency.

## Immediate execution checkpoint

Only Step 1 is active now.

The next implementation action is to add the AI service API and adapter around
the current deterministic text model. Do not implement model conversion,
Transformer layers, framebuffer code, or PS/2 input until Step 1 is built and
validated.
