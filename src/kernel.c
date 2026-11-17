#include "ai.h"
#include "ai_service.h"
#include "bench.h"
#include "kernel.h"
#include "llm_runtime.h"
#include "text_ai.h"
#include "ui.h"

/*
 * Boot orchestration. Everything here runs in ring 0 in a single address
 * space: there is no user/kernel boundary to cross and no scheduler to fight
 * for the CPU while inference runs.
 */

#define MULTIBOOT_BOOTLOADER_MAGIC 0x36D76289u
#define MULTIBOOT_TAG_END 0u
#define MULTIBOOT_TAG_MODULE 3u
#define BENCH_PASSES 10000u
#define SHELL_LINE_CAPACITY 80u

#define STAGE_TOTAL 8u

typedef struct {
    uint32_t total_size;
    uint32_t reserved;
} multiboot_info_t;

typedef struct {
    uint32_t type;
    uint32_t size;
} multiboot_tag_t;

typedef struct {
    uint32_t type;
    uint32_t size;
    uint32_t start;
    uint32_t end;
    char command_line[];
} multiboot_module_t;

static int bounded_string_equals(const char *text, size_t capacity,
                                 const char *expected)
{
    size_t i = 0;

    while (i < capacity && expected[i] != '\0' && text[i] == expected[i]) {
        ++i;
    }
    return i < capacity && expected[i] == '\0' && text[i] == '\0';
}

static int find_module(uint32_t info_address, const char *name,
                       uint64_t *start, uint32_t *bytes)
{
    const multiboot_info_t *info =
        (const multiboot_info_t *)(uintptr_t)info_address;
    uint32_t offset = sizeof(*info);

    while (offset + sizeof(multiboot_tag_t) <= info->total_size) {
        const multiboot_tag_t *tag =
            (const multiboot_tag_t *)((const uint8_t *)info + offset);

        if (tag->size < sizeof(*tag) || tag->size > info->total_size - offset) {
            return 0;
        }
        if (tag->type == MULTIBOOT_TAG_END) {
            break;
        }
        if (tag->type == MULTIBOOT_TAG_MODULE &&
            tag->size >= sizeof(multiboot_module_t)) {
            const multiboot_module_t *module =
                (const multiboot_module_t *)tag;
            size_t command_capacity = tag->size - sizeof(multiboot_module_t);

            if (bounded_string_equals(module->command_line, command_capacity,
                                      name) &&
                module->end >= module->start &&
                module->end <= (1u << 30)) {
                *start = module->start;
                *bytes = module->end - module->start;
                return 1;
            }
        }
        offset = (offset + tag->size + 7u) & ~7u;
    }
    return 0;
}

static int validate_module_ranges(uint32_t info_address)
{
    const multiboot_info_t *info =
        (const multiboot_info_t *)(uintptr_t)info_address;
    uint32_t starts[16];
    uint32_t ends[16];
    uint32_t count = 0;
    uint32_t offset = sizeof(*info);

    while (offset + sizeof(multiboot_tag_t) <= info->total_size) {
        const multiboot_tag_t *tag =
            (const multiboot_tag_t *)((const uint8_t *)info + offset);

        if (tag->size < sizeof(*tag) ||
            tag->size > info->total_size - offset) {
            return 0;
        }
        if (tag->type == MULTIBOOT_TAG_END) {
            break;
        }
        if (tag->type == MULTIBOOT_TAG_MODULE &&
            tag->size >= sizeof(multiboot_module_t)) {
            const multiboot_module_t *module =
                (const multiboot_module_t *)tag;
            uint32_t index;

            if (module->end < module->start ||
                module->end > (1u << 30) ||
                count >= sizeof(starts) / sizeof(starts[0])) {
                return 0;
            }
            for (index = 0; index < count; ++index) {
                if (module->start < ends[index] &&
                    starts[index] < module->end) {
                    return 0;
                }
            }
            starts[count] = module->start;
            ends[count] = module->end;
            ++count;
        }
        offset = (offset + tag->size + 7u) & ~7u;
    }
    return 1;
}

static void print_banner(void)
{
    serial_write("\n");
    serial_write("==============================================================================\n");
    serial_write("  BARE-METAL AI INFERENCE UNIKERNEL\n");
    serial_write("  x86-64 long mode | ring 0 | single address space | no scheduler\n");
    serial_write("==============================================================================\n\n");
}

static const char *skip_spaces(const char *text)
{
    while (*text == ' ') {
        ++text;
    }
    return text;
}

static void print_footprint(uint32_t kernel_bytes, uint32_t model_bytes,
                            uint32_t text_model_bytes, uint32_t samples_bytes,
                            uint32_t llm_model_bytes, uint32_t tokenizer_bytes,
                            size_t llm_arena_bytes,
                            uint64_t boot_cycles)
{
    serial_write("\n");
    serial_rule();
    serial_write("  FOOTPRINT\n");
    serial_rule();
    serial_write("  Kernel image in RAM .... ");
    serial_printf("%u B (%u KiB)\n", kernel_bytes, (kernel_bytes + 1023u) / 1024u);
    serial_write("  Kernel heap ............ ");
    serial_printf("%u B (%u KiB)\n", (uint32_t)kmalloc_capacity(),
                  (uint32_t)(kmalloc_capacity() / 1024u));
    serial_write("  Model weights .......... ");
    serial_printf("%u B (%u KiB)\n", model_bytes, (model_bytes + 1023u) / 1024u);
    serial_write("  Text model ............. ");
    serial_printf("%u B (%u KiB)\n", text_model_bytes,
                  (text_model_bytes + 1023u) / 1024u);
    serial_write("  Sample set ............. ");
    serial_printf("%u B (%u KiB)\n", samples_bytes,
                  (samples_bytes + 1023u) / 1024u);
    if (llm_model_bytes != 0u || tokenizer_bytes != 0u) {
        serial_write("  LLM model .............. ");
        serial_printf("%u B (%u KiB)\n", llm_model_bytes,
                      (llm_model_bytes + 1023u) / 1024u);
        serial_write("  Tokenizer .............. ");
        serial_printf("%u B (%u KiB)\n", tokenizer_bytes,
                      (tokenizer_bytes + 1023u) / 1024u);
        serial_write("  LLM arena .............. ");
        serial_printf("%u B (%u KiB)\n", (uint32_t)llm_arena_bytes,
                      (uint32_t)((llm_arena_bytes + 1023u) / 1024u));
    }
    serial_write("  Total resident ......... ");
    serial_printf("%u KiB\n",
                  (kernel_bytes + model_bytes + text_model_bytes +
                   samples_bytes + llm_model_bytes + tokenizer_bytes +
                   (uint32_t)llm_arena_bytes + 1023u) / 1024u);
    serial_write("  First inference complete . ");
    serial_printf("%llu cycles\n", (unsigned long long)boot_cycles);
}

static int text_equals(const char *left, const char *right)
{
    size_t i = 0;

    while (left[i] != '\0' && right[i] != '\0' && left[i] == right[i]) {
        ++i;
    }
    return left[i] == '\0' && right[i] == '\0';
}

static int text_starts_with(const char *text, const char *prefix)
{
    size_t index = 0;

    while (prefix[index] != '\0') {
        if (text[index] != prefix[index]) {
            return 0;
        }
        ++index;
    }
    return text[index] == '\0' || text[index] == ' ';
}

static int parse_uint32(const char *text, uint32_t *value)
{
    uint32_t result = 0;
    int digits = 0;

    while (*text == ' ') {
        ++text;
    }
    while (*text >= '0' && *text <= '9') {
        uint32_t digit = (uint32_t)(*text - '0');

        if (result > (UINT32_MAX - digit) / 10u) {
            return 0;
        }
        result = result * 10u + digit;
        ++text;
        digits = 1;
    }
    while (*text == ' ') {
        ++text;
    }
    if (*text != '\0' || !digits) {
        return 0;
    }
    *value = result;
    return 1;
}

static void print_shell_help(void)
{
    serial_write("\n  COMMANDS\n");
    serial_write("  help       show this command list\n");
    serial_write("  status     show kernel, memory, and selected-kernel status\n");
    serial_write("  predict N   classify frozen sample N (0-based) in ring 0\n");
    serial_write("  ask TEXT    generate a response with the kernel text model\n");
    serial_write("  run        run the benchmark again\n");
    serial_write("  clear      clear compatible serial terminals\n");
    serial_write("  shutdown   close QEMU and halt the kernel\n");
    serial_write("\n");
}

static int print_text_token(const char *text, size_t bytes,
                            uint32_t token_index, void *context)
{
    (void)token_index;
    (void)context;
    for (size_t index = 0; index < bytes; ++index) {
        serial_printf("%c", text[index]);
    }
    return 0;
}

static int print_llm_token(const uint8_t *bytes, size_t length,
                           uint32_t token_index, void *context)
{
    (void)token_index;
    (void)context;
    for (size_t index = 0; index < length; ++index) {
        serial_printf("%c", bytes[index]);
    }
    return 0;
}

static void print_text_response(ai_service_t *service, llm_runtime_t *llm,
                                const char *prompt)
{
    char response[AI_SERVICE_OUTPUT_CAPACITY];
    ai_generate_config_t config;
    ai_service_status_t status;
    size_t prompt_length = 0;

    if (*prompt == '\0') {
        serial_write("  usage: ask TEXT\n");
        return;
    }
    while (prompt[prompt_length] != '\0' &&
           prompt_length < TEXT_AI_PROMPT_LIMIT) {
        ++prompt_length;
    }
    if (prompt[prompt_length] != '\0') {
        serial_write("  ask failed: prompt is too long\n");
        return;
    }
    config.max_tokens = TEXT_AI_OUTPUT_LIMIT;
    config.output = response;
    config.output_capacity = sizeof(response);
    serial_write("  ai> ");
    if (llm != 0) {
        llm_status_t llm_status = llm_generate(
            llm, prompt, prompt_length, 64u, print_llm_token, 0);

        if (llm_status != LLM_OK) {
            serial_write("\n  ask failed: ");
            serial_write(llm_status_string(llm_status));
            serial_write("\n");
            return;
        }
        {
            llm_metrics_t metrics;

            if (llm_get_metrics(llm, &metrics) == LLM_OK) {
                serial_printf("\n  LLM metrics: %u prompt tokens, %u generated, "
                              "%llu cycles\n",
                              metrics.prompt_tokens, metrics.generated_tokens,
                              (unsigned long long)metrics.generation_cycles);
            }
        }
        serial_write("\n");
        return;
    }
    status = ai_generate_stream(service, prompt, prompt_length, &config,
                                print_text_token, 0);
    if (status != AI_SERVICE_OK) {
        serial_write("\n  ask failed: ");
        serial_write(ai_service_status_string(status));
        serial_write("\n");
        return;
    }
    if (response[0] == '\0' ||
        response[serial_length(response) - 1u] != '\n') {
        serial_write("\n");
    }
}

static void print_prediction(const ai_model_t *model, const ai_sample_t *samples,
                             uint32_t sample_count, ai_kernel_id_t kernel,
                             const char *argument)
{
    uint32_t index;
    float output[AI_OUTPUTS];
    uint32_t prediction;

    if (!parse_uint32(argument, &index) || index >= sample_count) {
        serial_printf("  usage: predict N (N must be 0..%u)\n",
                      sample_count == 0 ? 0u : sample_count - 1u);
        return;
    }

    ai_infer_kernel(model, samples[index].pixels, output, kernel);
    prediction = ai_argmax(output);
    serial_printf("  sample %u: predicted=%u, label=%u, %s\n",
                  index, prediction, samples[index].label,
                  prediction == samples[index].label ? "correct" : "different");
    serial_write("  logits: ");
    for (uint32_t output_index = 0; output_index < AI_OUTPUTS; ++output_index) {
        serial_printf("%s%u=%d.%06u", output_index == 0 ? "" : " ",
                      output_index, (int)output[output_index],
                      (uint32_t)((output[output_index] < 0.0f ?
                                  -output[output_index] :
                                  output[output_index]) * 1000000.0f) % 1000000u);
    }
    serial_write("\n");
}

static void print_shell_status(uint32_t kernel_bytes, uint32_t model_bytes,
                               uint32_t text_model_bytes,
                               uint32_t samples_bytes,
                               uint32_t sample_count,
                               ai_kernel_id_t selected_kernel)
{
    serial_write("\n  KERNEL STATUS\n");
    serial_write("  mode ................ interactive ring-0 shell\n");
    serial_printf("  kernel image ......... %u B\n", kernel_bytes);
    serial_printf("  model ................ %u B\n", model_bytes);
    serial_printf("  text model ........... %u B\n", text_model_bytes);
    serial_printf("  samples .............. %u records, %u B module\n",
                  sample_count, samples_bytes);
    serial_printf("  heap ................. %u / %u B\n",
                  (uint32_t)kmalloc_used(), (uint32_t)kmalloc_capacity());
    serial_printf("  free frames .......... %u\n", pmm_free_frames());
    serial_printf("  execution kernel ..... %s\n",
                  ai_kernel_name(selected_kernel));
    serial_write("\n");
}

static void shell_run(const ai_model_t *model,
                      ai_service_t *text_service,
                      llm_runtime_t *llm,
                      const ai_sample_t *samples,
                      uint32_t sample_count, const bench_config_t *config,
                      ai_kernel_id_t selected_kernel,
                      uint32_t kernel_bytes, uint32_t model_bytes,
                      uint32_t text_model_bytes,
                      uint32_t samples_bytes, bench_result_t *result)
{
    char line[SHELL_LINE_CAPACITY];
    uint32_t length = 0;

    serial_write("\n");
    serial_rule();
    serial_write("  INTERACTIVE KERNEL SHELL\n");
    serial_write("  Type 'help' for commands. The kernel stays resident until 'shutdown'.\n");
    serial_write("  kernel> ");

    for (;;) {
        if (serial_available()) {
            char character = serial_read();

            if (character == '\r' || character == '\n') {
                line[length] = '\0';
                serial_write("\n");
                if (text_equals(line, "help")) {
                    print_shell_help();
                } else if (text_equals(line, "status")) {
                    print_shell_status(kernel_bytes, model_bytes,
                                       text_model_bytes, samples_bytes,
                                       sample_count, selected_kernel);
                } else if (length >= 7u &&
                           line[0] == 'p' && line[1] == 'r' &&
                           line[2] == 'e' && line[3] == 'd' &&
                           line[4] == 'i' && line[5] == 'c' &&
                           line[6] == 't') {
                    if (line[7] != '\0' && line[7] != ' ') {
                        serial_write("  usage: predict N\n");
                    } else {
                        print_prediction(model, samples, sample_count,
                                         selected_kernel, line + 7);
                    }
                } else if (text_starts_with(line, "ask")) {
                    print_text_response(text_service, llm,
                                        skip_spaces(line + 3));
                } else if (text_equals(line, "run")) {
                    bench_run(model, samples, sample_count, config, result);
                    serial_printf("  benchmark complete: %u/%u correct\n",
                                  result->correct, result->samples);
                } else if (text_equals(line, "clear")) {
                    serial_write("\033[2J\033[H");
                } else if (text_equals(line, "shutdown") ||
                           text_equals(line, "exit")) {
                    serial_write("  shutting down.\n");
                    return;
                } else if (length != 0) {
                    serial_write("  unknown command; type 'help'.\n");
                }
                length = 0;
                serial_write("  kernel> ");
            } else if (character == '\b' || character == 127) {
                if (length != 0) {
                    --length;
                    serial_write("\b \b");
                }
            } else if (character >= 32 && character <= 126 &&
                       length + 1u < SHELL_LINE_CAPACITY) {
                line[length++] = character;
                serial_printf("%c", character);
            }
        }
    }
}

void kernel_main(uint32_t multiboot_info)
{
    const ai_model_t *model;
    const ai_sample_t *samples;
    uint64_t model_start;
    uint64_t samples_start;
    uint32_t model_bytes = 0;
    uint32_t text_model_bytes = 0;
    uint32_t llm_model_bytes = 0;
    uint32_t tokenizer_bytes = 0;
    uint32_t samples_bytes = 0;
    uint32_t dashboard_bytes = 0;
    uint32_t sample_count = 0;
    uint32_t kernel_bytes = (uint32_t)(_kernel_end - _kernel_start);
    uint64_t tsc_at_entry = read_tsc();
    uint64_t cycles_per_second;
    uint64_t nanoseconds_per_cycle_picos;
    ai_caps_t caps;
    ai_dispatch_record_t dispatch;
    ai_kernel_id_t selected_kernel;
    uint32_t k;
    bench_config_t config;
    bench_result_t result;
    uint32_t frames_before_test;
    uint64_t test_frame;
    uint64_t dashboard_start;
    int dashboard_mode;
    uint64_t llm_model_start;
    uint64_t tokenizer_start;
    int llm_present;
    ui_console_t console;
    int framebuffer_available;
    static uint8_t ai_arena[AI_SERVICE_ARENA_BYTES]
        __attribute__((aligned(AI_SERVICE_ARENA_ALIGNMENT)));
    ai_service_t text_service;
    ai_service_status_t ai_status;
    static uint8_t llm_arena[LLM_RUNTIME_ARENA_BYTES]
        __attribute__((aligned(64)));
    llm_runtime_t llm_runtime;

    serial_init();
    print_banner();

    serial_aligned("[1/8] Serial console (COM1, 115200 8N1)", "READY");

    idt_init();
    serial_aligned("[2/8] IDT: 32 CPU exception gates (vectors 0-31)", "READY");

    if (!pmm_init(multiboot_info)) {
        kernel_panic("Multiboot2 memory map missing or invalid");
    }
    frames_before_test = pmm_free_frames();
    test_frame = pmm_alloc_frame();
    if (test_frame == 0 || pmm_free_frames() + 1u != frames_before_test ||
        !pmm_free_frame(test_frame) ||
        pmm_free_frames() != frames_before_test) {
        kernel_panic("physical frame allocator smoke test failed");
    }
    serial_aligned("[3/8] Physical frame allocator (4 KiB bitmap)", "READY");
    serial_printf("        usable RAM %llu KiB | free frames %u\n",
                  (unsigned long long)(pmm_usable_bytes() / 1024u),
                  pmm_free_frames());

    if (kmalloc(32u) == 0) {
        kernel_panic("kernel heap smoke test failed");
    }
    if (!vmm_map_2m(0x200000u, 0x200000u, 0x3u)) {
        kernel_panic("2 MiB page mapping smoke test failed");
    }
    serial_aligned("[4/8] Paging: 1 GiB identity map, 2 MiB pages", "READY");
    serial_printf("        heap %u B (bump allocator) | mapped %u MiB\n",
                  (uint32_t)kmalloc_capacity(), 1024u);

    if (!validate_module_ranges(multiboot_info)) {
        kernel_panic("Multiboot2 module ranges are invalid or overlap");
    }
    if (!find_module(multiboot_info, "model", &model_start, &model_bytes)) {
        kernel_panic("GRUB module 'model' was not found");
    }
    model = (const ai_model_t *)(uintptr_t)model_start;
    if (model_bytes != sizeof(ai_model_t) ||
        !ai_model_valid(model, model_bytes)) {
        kernel_panic("model module has an invalid format or size");
    }

    if (!find_module(multiboot_info, "samples", &samples_start,
                     &samples_bytes)) {
        kernel_panic("GRUB module 'samples' was not found");
    }
    samples = (const ai_sample_t *)(uintptr_t)(samples_start +
                                              sizeof(ai_sample_header_t));
    if (!ai_samples_valid((const void *)(uintptr_t)samples_start, samples_bytes,
                          &sample_count)) {
        kernel_panic("sample module has an invalid format or size");
    }
    serial_aligned("[5/8] GRUB modules (initrd, no filesystem)", "READY");
    serial_printf("        mlp %u -> %u -> %u | model %u B | samples %u x %u B\n",
                  AI_INPUTS, AI_HIDDEN, AI_OUTPUTS, model_bytes, sample_count,
                  (uint32_t)sizeof(ai_sample_t));

    if (!find_module(multiboot_info, "text-model", &model_start,
                     &text_model_bytes)) {
        kernel_panic("GRUB module 'text-model' was not found");
    }
    ai_status = ai_init_arena(&text_service, ai_arena, sizeof(ai_arena));
    if (ai_status != AI_SERVICE_OK) {
        kernel_panic("text AI service arena initialization failed");
    }
    ai_status = ai_load_model(&text_service, (const void *)(uintptr_t)model_start,
                              text_model_bytes);
    if (ai_status != AI_SERVICE_OK) {
        kernel_panic("text model module has an invalid format or size");
    }
    llm_present = find_module(multiboot_info, "llm-model", &llm_model_start,
                              &llm_model_bytes);
    if (find_module(multiboot_info, "tokenizer", &tokenizer_start,
                    &tokenizer_bytes) != llm_present) {
        kernel_panic("LLM model and tokenizer modules must be paired");
    }
    if (llm_present) {
        llm_status_t llm_status = llm_runtime_init(
            &llm_runtime, llm_arena, sizeof(llm_arena),
            (const void *)(uintptr_t)llm_model_start, llm_model_bytes,
            (const void *)(uintptr_t)tokenizer_start, tokenizer_bytes);

        if (llm_status != LLM_OK) {
            kernel_panic("LLM model or tokenizer is invalid");
        }
        serial_printf("        LLM package ........ %u B model | %u B tokenizer\n",
                      llm_model_bytes, tokenizer_bytes);
        serial_printf("        LLM validation ..... %llu cycles | arena %u B\n",
                      (unsigned long long)llm_runtime.metrics.validation_cycles,
                      (uint32_t)llm_runtime.metrics.arena_used);
    }
    framebuffer_available = ui_init(multiboot_info, &console);
    if (framebuffer_available) {
        serial_write(console.text_mode ?
                     "        VGA text console .... ready\n" :
                     "        framebuffer ........ ready\n");
    } else {
        serial_write("        framebuffer ........ unavailable; serial fallback\n");
        if (console.framebuffer_address != 0u) {
            serial_printf("        framebuffer tag ..... %llx, %ux%u, pitch %u, bpp %u\n",
                          (unsigned long long)console.framebuffer_address,
                          console.width, console.height,
                          console.framebuffer_pitch, console.framebuffer_bpp);
        }
    }
    dashboard_mode = find_module(multiboot_info, "dashboard",
                                 &dashboard_start, &dashboard_bytes) &&
                     dashboard_bytes != 0;

    cycles_per_second = tsc_calibrate();
    if (cycles_per_second != 0) {
        serial_aligned("[6/8] TSC calibration (PIT channel 2, 50 ms)", "READY");
        nanoseconds_per_cycle_picos =
            (1000000000000ULL / cycles_per_second);
        serial_printf("        %llu cycles/s | %llu.%03llu ns per cycle\n",
                      (unsigned long long)cycles_per_second,
                      (unsigned long long)(nanoseconds_per_cycle_picos / 1000u),
                      (unsigned long long)(nanoseconds_per_cycle_picos % 1000u));
    } else {
        serial_aligned("[6/8] TSC calibration (PIT channel 2, 50 ms)",
                       "UNAVAILABLE");
        serial_write("        Reporting raw cycles; no converted timings.\n");
    }

    serial_aligned("[7/8] CPU features (CPUID snapshot)", "READY");
    ai_probe_caps(&caps);
    if (serial_length(skip_spaces(caps.brand)) != 0) {
        serial_printf("        %s\n", skip_spaces(caps.brand));
    }
    serial_printf("        avx=%d avx2=%d fma=%d avx_state=%d\n",
                  caps.has_avx, caps.has_avx2, caps.has_fma,
                  caps.avx_state_ready);

    serial_write("\n");
    serial_rule();
    serial_write("  DISPATCH (measured, not assumed)\n");
    serial_rule();
    selected_kernel = ai_dispatch(model, ai_dispatch_input(), &caps,
                                  read_tsc, &dispatch);
    serial_printf("  selected ......... %s\n",
                  ai_kernel_name(selected_kernel));
    serial_printf("  eligible ......... 0x%x\n", dispatch.eligible_mask);
    for (k = 0; k < AI_KERNEL_COUNT; ++k) {
        const char *name = ai_kernel_name((ai_kernel_id_t)k);

        if ((dispatch.measured_mask & (1u << k)) == 0 &&
            (dispatch.eligible_mask & (1u << k)) == 0) {
            continue;
        }
        serial_write("  ");
        serial_write(name);
        serial_write(" ");
        if (serial_length(name) < 16u) {
            serial_repeat('.', 16u - (uint32_t)serial_length(name));
        }
        serial_write(" ");
        if ((dispatch.measured_mask & (1u << k)) != 0) {
            serial_printf("%llu cycles",
                          (unsigned long long)dispatch.cycles[k]);
            if (k != AI_KERNEL_SCALAR) {
                serial_printf(" | diff %u", dispatch.diff[k]);
            }
        } else {
            serial_write("not measured");
        }
        if ((dispatch.banned_mask & (1u << k)) != 0) {
            serial_write(" | FAILED gate");
        }
        serial_write("\n");
    }
    serial_aligned("[8/8] Execution kernel selected by measurement", "READY");
    serial_printf("        code path ...... %s\n",
                  ai_kernel_name(selected_kernel));

    config.passes = BENCH_PASSES;
    config.kernel_bytes = kernel_bytes;
    config.model_bytes = model_bytes;
    config.samples_bytes = samples_bytes;
    config.free_frames = pmm_free_frames();
    config.usable_bytes = pmm_usable_bytes();
    config.cycles_per_second = cycles_per_second;
    config.kernel_entry_cycles = tsc_at_entry;
    config.caps = caps;
    config.dispatch = dispatch;

    bench_run(model, samples, sample_count, &config, &result);
    print_footprint(kernel_bytes, model_bytes, text_model_bytes, samples_bytes,
                    llm_model_bytes, tokenizer_bytes,
                    llm_present ? llm_runtime.metrics.arena_used : 0u,
                    result.first_inference_cycles);

    serial_write("\n");
    serial_write("==============================================================================\n");
    serial_aligned("RUN COMPLETE", "interactive shell ready");
    serial_write("==============================================================================\n");

    if (dashboard_mode) {
        serial_write("  dashboard capture complete; shutting down.\n");
    } else if (llm_present && framebuffer_available) {
        serial_write("  framebuffer UI ready; press Escape to halt.\n");
        keyboard_init();
        __asm__ volatile ("sti");
        ui_write(&console, (const uint8_t *)"> ", 2u);
        while (console.running) {
            ui_keyboard_poll(&console, &llm_runtime);
            __asm__ volatile ("hlt");
        }
        __asm__ volatile ("cli");
        serial_printf("  framebuffer render cycles: %llu\n",
                      (unsigned long long)console.render_cycles);
        serial_printf("  LLM generation cycles: %llu\n",
                      (unsigned long long)llm_runtime.metrics.generation_cycles);
    } else {
        shell_run(model, &text_service, llm_present ? &llm_runtime : 0,
                  samples, sample_count, &config,
                  selected_kernel, kernel_bytes, model_bytes, text_model_bytes,
                  samples_bytes, &result);
    }
    qemu_debug_exit(0);
    for (;;) {
        __asm__ volatile ("cli; hlt");
    }
}
