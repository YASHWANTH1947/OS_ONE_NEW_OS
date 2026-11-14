#include <stdint.h>

#include "ai.h"

/*
 * AVX2 + FMA kernel, compiled with -mavx2 -mfma. Same freestanding
 * conventions as kernels_avx2.c. The fused multiply-add chain starts with the
 * bias, folds every product into the accumulator, and produces one final
 * rounding per partial-sum lane.
 *
 * Eligibility (CPUID FMA + enabled AVX state) is the dispatcher's job; the
 * correctness gate must pass before this kernel can be selected anywhere.
 */

typedef float v8 __attribute__((vector_size(32), may_alias));

/* Unaligned 8-float load; see kernels_avx2.c for why this is a memcpy and
 * not a cast (vmovaps faults on unaligned rows; memcpy lowers to vmovups). */
static inline v8 vload(const float *p)
{
    v8 out;

    __builtin_memcpy(&out, p, sizeof(out));
    return out;
}

static inline v8 vfma(v8 a, v8 b, v8 c)
{
    return __builtin_ia32_vfmaddps256(a, b, c);
}

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

void ai_forward_fma(const ai_model_t *model, const float input[AI_INPUTS],
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
            acc = vfma(vload(&w[i]), vload(&input[i]), acc);
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

        for (i = 0; i + 7u < AI_HIDDEN; i += 8u) {
            acc = vfma(vload(&w[i]), vload(&hidden[i]), acc);
        }
        for (; i < AI_HIDDEN; ++i) {
            tail += w[i] * hidden[i];
        }
        sum = vsum8(acc) + tail;
        output[o] = sum;
    }
}
