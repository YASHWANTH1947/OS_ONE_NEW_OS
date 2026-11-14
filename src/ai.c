#include "ai.h"

/*
 * AEGIS-X System B: the adaptive execution engine core.
 *
 * Compiled for every environment (unikernel, host baseline, unit tests) with
 * portable flags. Owns package validation, the kernel registry, the measured
 * dispatch policy, and the correctness gate. The per-kernel loops live in
 * separate translation units with their own ISA flags (kernels_scalar.c,
 * kernels_avx2.c, kernels_fma.c), so this file contains no YMM instructions.
 */

/* ------------------------------------------------------------------ data -- */
/* Deterministic calibration input for dispatch timing, matching the workload's
 * value range (MNIST-normalised intensities). The pattern is fixed in source
 * so the dispatch decision is reproducible from source alone. */
static float dispatch_input[AI_INPUTS];

static void fill_dispatch_input(void)
{
    uint32_t state = 0xC0FFEEu;
    uint32_t i;

    for (i = 0; i < AI_INPUTS; ++i) {
        state = state * 1664525u + 1013904223u;
        dispatch_input[i] = (float)(30u + ((state >> 8) & 0x7Fu)) / 255.0f;
    }
}

const float *ai_dispatch_input(void)
{
    static int primed = 0;

    if (!primed) {
        fill_dispatch_input();
        primed = 1;
    }
    return dispatch_input;
}

/* --------------------------------------------------------------- naming -- */

const char *ai_kernel_name(ai_kernel_id_t kernel)
{
    switch (kernel) {
    case AI_KERNEL_SCALAR:
        return "scalar";
    case AI_KERNEL_AVX2:
        return "avx2";
    case AI_KERNEL_FMA:
        return "fma";
    default:
        return "unknown";
    }
}

/* ------------------------------------------------------------ validation -- */

int ai_model_valid(const ai_model_t *model, size_t bytes)
{
    if (model == 0 || bytes < sizeof(ai_model_t)) {
        return 0;
    }

    return model->magic == AI_MAGIC &&
           model->version == AI_VERSION &&
           model->inputs == AI_INPUTS &&
           model->hidden == AI_HIDDEN &&
           model->outputs == AI_OUTPUTS;
}

int ai_samples_valid(const void *buffer, size_t bytes, uint32_t *count)
{
    const ai_sample_header_t *header = (const ai_sample_header_t *)buffer;
    uint64_t expected;

    if (buffer == 0 || bytes < sizeof(*header)) {
        return 0;
    }
    if (header->magic != AI_SAMPLE_MAGIC ||
        header->version != AI_SAMPLE_VERSION ||
        header->reserved != 0 ||
        header->count == 0 || header->count > AI_SAMPLE_LIMIT) {
        return 0;
    }
    expected = (uint64_t)sizeof(*header) +
               (uint64_t)header->count * sizeof(ai_sample_t);
    if (expected != bytes) {
        return 0;
    }
    {
        const ai_sample_t *samples =
            (const ai_sample_t *)((const uint8_t *)buffer + sizeof(*header));
        uint32_t sample;

        for (sample = 0; sample < header->count; ++sample) {
            uint32_t pixel;

            if (samples[sample].label >= AI_OUTPUTS) {
                return 0;
            }
            for (pixel = 0; pixel < AI_INPUTS; ++pixel) {
                float value = samples[sample].pixels[pixel];

                if ( value != value || value < 0.0f || value > 1.0f) {
                    return 0;
                }
            }
        }
    }
    if (count != 0) {
        *count = header->count;
    }
    return 1;
}

/* ------------------------------------------------------------ numerics -- */

static float absf(float value)
{
    union { float f; uint32_t u; } bits;

    bits.f = value;
    bits.u &= 0x7FFFFFFFu;
    return bits.f;
}

uint32_t ai_output_diff(const float reference[AI_OUTPUTS],
                        const float candidate[AI_OUTPUTS])
{
    float worst = 0.0f;
    uint32_t i;

    for (i = 0; i < AI_OUTPUTS; ++i) {
        float delta = absf(reference[i] - candidate[i]);

        if (delta > worst) {
            worst = delta;
        }
    }
    /* Small magnitudes keep this conversion inside cvttss2si range. */
    return (uint32_t)(worst * (float)AI_DIFF_SCALE);
}

uint32_t ai_argmax(const float values[AI_OUTPUTS])
{
    uint32_t best = 0;
    uint32_t i;

    for (i = 1; i < AI_OUTPUTS; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

/* -------------------------------------------------------------- registry -- */

void ai_infer_kernel(const ai_model_t *model, const float input[AI_INPUTS],
                     float output[AI_OUTPUTS], ai_kernel_id_t kernel)
{
    switch (kernel) {
    case AI_KERNEL_SCALAR:
        ai_forward_scalar(model, input, output);
        return;
    case AI_KERNEL_AVX2:
        ai_forward_avx2(model, input, output);
        return;
    case AI_KERNEL_FMA:
        ai_forward_fma(model, input, output);
        return;
    default:
        /* Unreachable without memory corruption; play safe, not fast. */
        ai_forward_scalar(model, input, output);
        return;
    }
}

void ai_infer(const ai_model_t *model, const float input[AI_INPUTS],
              float output[AI_OUTPUTS])
{
    ai_infer_kernel(model, input, output, AI_KERNEL_SCALAR);
}

/* --------------------------------------------------------------- timing -- */
/* Timing a single pass is too noisy under TCG; the dispatch calibration
 * times a small batch and divides, still by reading the environment's cycle
 * source around the WHOLE batch (so the reading overhead is amortised rather
 * than added per pass). */

#define DISPATCH_WARMUP 8u
#define DISPATCH_BATCH 16u

/* Eligibility is a pure function of the capability snapshot. The scalar
 * kernel is always eligible; SIMD kernels additionally require the AVX
 * state machine to have been enabled by the environment layer (a VEX
 * instruction with XCR0.AVX clear raises an exception regardless of what
 * CPUID advertises). */
int ai_kernel_eligible(ai_kernel_id_t kernel, const ai_caps_t *caps)
{
    switch (kernel) {
    case AI_KERNEL_SCALAR:
        return 1;
    case AI_KERNEL_AVX2:
        return caps != 0 && caps->avx_state_ready && caps->has_avx2;
    case AI_KERNEL_FMA:
        return caps != 0 && caps->avx_state_ready && caps->has_avx2 &&
               caps->has_fma;
    default:
        return 0;
    }
}

uint64_t ai_time_kernel(const ai_model_t *model, const float input[AI_INPUTS],
                        float output[AI_OUTPUTS], ai_kernel_id_t kernel,
                        ai_cycles_fn cycles_fn)
{
    uint64_t best = UINT64_MAX;
    uint32_t trial;
    uint32_t i;

    for (i = 0; i < DISPATCH_WARMUP; ++i) {
        ai_infer_kernel(model, input, output, kernel);
    }
    for (trial = 0; trial < 3u; ++trial) {
        uint64_t cycles;

        cycles = cycles_fn();
        for (i = 0; i < DISPATCH_BATCH; ++i) {
            ai_infer_kernel(model, input, output, kernel);
        }
        cycles = cycles_fn() - cycles;
        if (cycles / DISPATCH_BATCH < best) {
            best = cycles / DISPATCH_BATCH;
        }
    }
    return best == UINT64_MAX ? 0u : best;
}

/* --------------------------------------------------------------- dispatch -- */

ai_kernel_id_t ai_dispatch(const ai_model_t *model,
                           const float input[AI_INPUTS],
                           const ai_caps_t *caps, ai_cycles_fn cycles_fn,
                           ai_dispatch_record_t *record)
{
    float reference[AI_OUTPUTS];
    float candidate[AI_OUTPUTS];
    float timing_scratch[AI_OUTPUTS];
    ai_kernel_id_t best_kernel = AI_KERNEL_SCALAR;
    uint64_t best_cycles;
    uint32_t kernel;

    if (record == 0) {
        return AI_KERNEL_SCALAR;
    }
    record->selected = AI_KERNEL_SCALAR;
    record->eligible_mask = 0u;
    record->measured_mask = 0u;
    record->banned_mask = 0u;
    for (kernel = 0; kernel < AI_KERNEL_COUNT; ++kernel) {
        record->cycles[kernel] = 0u;
        record->diff[kernel] = 0u;
    }

    if (model == 0 || caps == 0 || input == 0) {
        return AI_KERNEL_SCALAR;
    }

    /* Reference output from the portable kernel on the calibration input.
     * Timing uses a scratch buffer so the reference is never overwritten;
     * the gate then provably compares against the preserved value. */
    record->eligible_mask |= 1u << AI_KERNEL_SCALAR;
    ai_infer_kernel(model, input, reference, AI_KERNEL_SCALAR);
    record->diff[AI_KERNEL_SCALAR] = 0u;
    record->cycles[AI_KERNEL_SCALAR] =
        ai_time_kernel(model, input, timing_scratch, AI_KERNEL_SCALAR,
                       cycles_fn);
    record->measured_mask |= 1u << AI_KERNEL_SCALAR;

    best_cycles = record->cycles[AI_KERNEL_SCALAR];
    best_kernel = AI_KERNEL_SCALAR;

    for (kernel = 0; kernel < AI_KERNEL_COUNT; ++kernel) {
        if (kernel == AI_KERNEL_SCALAR) {
            continue;
        }
        if (!ai_kernel_eligible((ai_kernel_id_t)kernel, caps)) {
            continue;
        }
        record->eligible_mask |= 1u << kernel;
        ai_infer_kernel(model, input, candidate, (ai_kernel_id_t)kernel);
        record->diff[kernel] = ai_output_diff(reference, candidate);
        if (record->diff[kernel] > AI_KERNEL_GATE_DIFF) {
            record->banned_mask |= 1u << kernel;
            continue;
        }
        record->cycles[kernel] =
            ai_time_kernel(model, input, timing_scratch,
                           (ai_kernel_id_t)kernel, cycles_fn);
        record->measured_mask |= 1u << kernel;
        if (record->cycles[kernel] < best_cycles) {
            best_cycles = record->cycles[kernel];
            best_kernel = (ai_kernel_id_t)kernel;
        }
    }

    record->selected = best_kernel;
    return best_kernel;
}
