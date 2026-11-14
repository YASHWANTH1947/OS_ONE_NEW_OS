#include <stdint.h>

#include "ai.h"

/*
 * AVX2 kernel, compiled with -mavx2 (no FMA). 256-bit vector operations via
 * GCC vector extensions rather than immintrin.h, keeping the file fully
 * freestanding. Eligibility is checked by the dispatcher before this code is
 * ever called (CPUID AVX2 + enabled AVX state), so YMM state is guaranteed.
 *
 * Alignment: weight rows start at 4-byte granularity from an arbitrary model
 * base, so loads are unaligned (vmovups). The dispatcher's correctness gate
 * compares the result against the scalar reference before this kernel may be
 * selected, and the benchmark passes model weights from a GRUB module whose
 * loading address is page-aligned by GRUB (typically 4 KiB or better).
 */

typedef float v8 __attribute__((vector_size(32), may_alias));

/* Unaligned 8-float load. Model/sample rows start at arbitrary 4-byte
 * addresses, and a plain cast-deref would emit vmovaps and fault (verified
 * empirically on this toolchain). __builtin_memcpy is lowered to a single
 * unaligned vmovups while remaining correct at any alignment. */
static inline v8 vload(const float *p)
{
    v8 out;

    __builtin_memcpy(&out, p, sizeof(out));
    return out;
}

/* Horizontal sum of the 8 lanes using same-width shuffles only. */
static inline float vsum8(v8 v)
{
    v8 h  = __builtin_shufflevector(v, v, 4, 5, 6, 7, 0, 1, 2, 3);
    v8 f1 = v + h;
    v8 g  = __builtin_shufflevector(f1, f1, 2, 3, 0, 1, 4, 5, 6, 7);
    v8 f2 = f1 + g;
    v8 k  = __builtin_shufflevector(f2, f2, 1, 0, 2, 3, 4, 5, 6, 7);
    v8 f3 = f2 + k;
    return f3[0];
}

void ai_forward_avx2(const ai_model_t *model, const float input[AI_INPUTS],
                     float output[AI_OUTPUTS])
{
    float hidden[AI_HIDDEN] __attribute__((aligned(32)));
    uint32_t h;
    uint32_t i;
    uint32_t o;

    for (h = 0; h < AI_HIDDEN; ++h) {
        const float *w = &model->w1[h * AI_INPUTS];
        v8 acc = {model->b1[h], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float tail = 0.0f;

        for (i = 0; i + 7u < AI_INPUTS; i += 8u) {
            acc += vload(&w[i]) * vload(&input[i]);
        }
        for (; i < AI_INPUTS; ++i) {
            tail += w[i] * input[i];
        }
        hidden[h] = vsum8(acc) + tail;
        if (hidden[h] < 0.0f) {
            hidden[h] = 0.0f;
        }
    }

    for (o = 0; o < AI_OUTPUTS; ++o) {
        const float *w = &model->w2[o * AI_HIDDEN];
        v8 acc = {model->b2[o], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float tail = 0.0f;
        float sum;

        /* AI_HIDDEN = 128 = 16 x 8 exactly, so no tail loop runs here. */
        for (i = 0; i + 7u < AI_HIDDEN; i += 8u) {
            acc += vload(&w[i]) * vload(&hidden[i]);
        }
        for (; i < AI_HIDDEN; ++i) {
            tail += w[i] * hidden[i];
        }
        sum = vsum8(acc) + tail;
        output[o] = sum;
    }
}
