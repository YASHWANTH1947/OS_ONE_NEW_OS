#ifndef AI_H
#define AI_H

#include <stddef.h>
#include <stdint.h>

#define AI_MAGIC 0x41494D4Cu
#define AI_VERSION 1u
#define AI_INPUTS 784u
#define AI_HIDDEN 128u
#define AI_OUTPUTS 10u

#define AI_SAMPLE_MAGIC 0x504D4153u /* "SAMP" */
#define AI_SAMPLE_VERSION 1u
#define AI_SAMPLE_LIMIT 4096u

/* ------------------------------------------------------------------------- */
/* Model and sample packages                                                 */
/* ------------------------------------------------------------------------- */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t inputs;
    uint32_t hidden;
    uint32_t outputs;
    float w1[AI_HIDDEN * AI_INPUTS];
    float b1[AI_HIDDEN];
    float w2[AI_OUTPUTS * AI_HIDDEN];
    float b2[AI_OUTPUTS];
} ai_model_t;

typedef struct {
    uint32_t label;
    float pixels[AI_INPUTS];
} ai_sample_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t reserved;
} ai_sample_header_t;

int ai_model_valid(const ai_model_t *model, size_t bytes);
int ai_samples_valid(const void *buffer, size_t bytes, uint32_t *count);

/* ------------------------------------------------------------------------- */
/* Dispatch engine (AEGIS-X System B)                                        */
/* ------------------------------------------------------------------------- */
/*
 * The kernel ships multiple mathematically equivalent implementations of the
 * forward pass and selects one at boot through a MEASURED policy, not a
 * CPUID assumption. The probe (probe/) established why: under this project's
 * own benchmark environment (QEMU TCG), AVX2 code is ~3.6x SLOWER than the
 * tuned scalar loop and FMA ~19.5x slower, despite CPUID advertising both
 * after -cpu max. A capability-based dispatcher would therefore make the
 * unikernel slower on its own benchmark; a measured dispatcher picks scalar
 * there and FMA on silicon, and records the evidence either way.
 *
 * Kernels live in separate translation units (src/kernels_scalar.c,
 * src/kernels_avx2.c, src/kernels_fma.c), each compiled with its own ISA
 * flags, so the portable kernel never contains a YMM instruction and stays
 * safe on hardware without an enabled AVX state.
 */

typedef enum {
    AI_KERNEL_SCALAR = 0,   /* tuned portable C, four-accumulator unroll     */
    AI_KERNEL_AVX2,         /* 256-bit loads/mults, separate add            */
    AI_KERNEL_FMA,          /* 256-bit fused multiply-add                   */
    AI_KERNEL_COUNT
} ai_kernel_id_t;

/* Human-readable kernel names, in ai_kernel_id_t order. */
const char *ai_kernel_name(ai_kernel_id_t kernel);

/* Feature snapshot taken once at boot by the environment layer
 * (src/cpuid.c in the unikernel; the same CPUID sequence in the host build).
 * Capabilities are prerequisites, not decisions: the measured dispatcher
 * decides which eligible kernel to run. */
typedef struct {
    int has_sse2;        /* x86-64 baseline; always 1 on this target         */
    int has_avx;         /* CPUID.1.ECX[28]                                  */
    int has_avx2;        /* CPUID.(7,0).EBX[5]                               */
    int has_fma;         /* CPUID.1.ECX[12]                                 */
    int avx_state_ready; /* CR4.OSXSAVE set and XCR0 enables x87+SSE+AVX     */
    char brand[49];      /* CPUID.0x80000002..4 brand string, NUL-terminated */
} ai_caps_t;

/* Fill caps with the current CPU's features and, where the executing
 * environment permits, enable the AVX state machine before testing it.
 * Returns 1 when the SIMD-relevant state is ready, 0 otherwise. */
int ai_probe_caps(ai_caps_t *caps);

/* Raw CPUID leaf access (environment-provided; src/cpuid.c in the unikernel,
 * the same sequence in the host build). Exposed for diagnostics and tests. */
void ai_raw_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);

/* Is a kernel eligible for measurement on this machine? Pure function of
 * caps; the scalar kernel is always eligible. */
int ai_kernel_eligible(ai_kernel_id_t kernel, const ai_caps_t *caps);

/* Maximum element-wise absolute difference between two forward-pass outputs,
 * scaled by AI_DIFF_SCALE (integer-only to keep it reportable everywhere). */
#define AI_DIFF_SCALE 1000000u
uint32_t ai_output_diff(const float reference[AI_OUTPUTS],
                        const float candidate[AI_OUTPUTS]);

/* Correctness gate for a non-scalar kernel: it must reproduce the scalar
 * reference within this scaled bound on the calibration input. The bound is
 * generous (reordered float32 sums on 784-element dot products), and the
 * probe measured an observed disagreement of ~15 units at this scale. */
#define AI_KERNEL_GATE_DIFF 500u

typedef uint64_t ai_cycles_fn(void);

/* Best-of-three timed cycles for a single forward pass with the given
 * kernel over the calibration input. */
uint64_t ai_time_kernel(const ai_model_t *model, const float input[AI_INPUTS],
                        float output[AI_OUTPUTS], ai_kernel_id_t kernel,
                        ai_cycles_fn cycles_fn);

/* Full dispatch record: what was eligible, what was measured, what passed
 * the correctness gate, what each cost, and what was selected. Every field
 * is serialised into the result block and the dashboard. */
typedef struct {
    ai_kernel_id_t selected;
    uint32_t eligible_mask;      /* bit i set when kernel i was eligible     */
    uint32_t measured_mask;      /* bit i set when kernel i was timed        */
    uint32_t banned_mask;        /* bit i set when kernel i failed the gate  */
    uint64_t cycles[AI_KERNEL_COUNT];
    uint32_t diff[AI_KERNEL_COUNT];  /* vs scalar reference, AI_DIFF_SCALE  */
} ai_dispatch_record_t;

/* Measured dispatch: probe capabilities, time every eligible kernel, gate
 * every non-scalar candidate against the scalar reference, and return the
 * fastest surviving kernel. Fills record with complete evidence. Never
 * selects a kernel that failed the gate, regardless of speed. */
ai_kernel_id_t ai_dispatch(const ai_model_t *model,
                           const float input[AI_INPUTS],
                           const ai_caps_t *caps, ai_cycles_fn cycles_fn,
                           ai_dispatch_record_t *record);

/* Forward pass with an explicit kernel, so any number the system reports can
 * be attributed to the code path that produced it. */
void ai_infer_kernel(const ai_model_t *model, const float input[AI_INPUTS],
                     float output[AI_OUTPUTS], ai_kernel_id_t kernel);

/* Portable reference implementation (the scalar kernel). */
void ai_infer(const ai_model_t *model, const float input[AI_INPUTS],
              float output[AI_OUTPUTS]);

uint32_t ai_argmax(const float values[AI_OUTPUTS]);

/* Deterministic calibration input used by the dispatch policy. */
const float *ai_dispatch_input(void);

/* Per-kernel forward passes, implemented in separate translation units with
 * their own ISA flags. Declared here so the registry can link them; callers
 * should use ai_infer_kernel rather than calling these directly. */
void ai_forward_scalar(const ai_model_t *model, const float input[AI_INPUTS],
                        float output[AI_OUTPUTS]);
void ai_forward_avx2(const ai_model_t *model, const float input[AI_INPUTS],
                      float output[AI_OUTPUTS]);
void ai_forward_fma(const ai_model_t *model, const float input[AI_INPUTS],
                     float output[AI_OUTPUTS]);

#endif
