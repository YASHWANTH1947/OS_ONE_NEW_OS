#include "ai.h"

/*
 * Portable scalar kernel: the reference implementation every other kernel is
 * gated against. No ISA flags beyond the x86-64 baseline (SSE2).
 *
 * The loop is the tuned four-accumulator form: four independent partial sums
 * give the CPU enough instruction-level parallelism to hide multiply
 * latency, and contiguous weight rows are prefetcher friendly at 3 KiB each.
 * The compiler may autovectorize this to SSE2, which is legal in the
 * unikernel because boot.S enables SSE2 state before kernel_main runs.
 */

void ai_forward_scalar(const ai_model_t *model, const float input[AI_INPUTS],
                       float output[AI_OUTPUTS])
{
    float hidden[AI_HIDDEN];
    uint32_t h;
    uint32_t i;
    uint32_t o;

    for (h = 0; h < AI_HIDDEN; ++h) {
        const float *w = &model->w1[h * AI_INPUTS];
        float s0 = model->b1[h];
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;

        for (i = 0; i + 3u < AI_INPUTS; i += 4u) {
            s0 += w[i] * input[i];
            s1 += w[i + 1u] * input[i + 1u];
            s2 += w[i + 2u] * input[i + 2u];
            s3 += w[i + 3u] * input[i + 3u];
        }
        for (; i < AI_INPUTS; ++i) {
            s0 += w[i] * input[i];
        }
        hidden[h] = (s0 + s1) + (s2 + s3);
        if (hidden[h] < 0.0f) {
            hidden[h] = 0.0f;
        }
    }

    for (o = 0; o < AI_OUTPUTS; ++o) {
        const float *w = &model->w2[o * AI_HIDDEN];
        float sum = model->b2[o];
        uint32_t h2;

        for (h2 = 0; h2 < AI_HIDDEN; ++h2) {
            sum += w[h2] * hidden[h2];
        }
        output[o] = sum;
    }
}
