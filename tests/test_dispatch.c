#define _POSIX_C_SOURCE 200809L

#include "ai.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Kernel-equivalence tests (AEGIS-X System B correctness gate, host side).
 *
 * Two claims must hold before the dispatcher may ever select a SIMD kernel:
 *   1. All eligible kernels agree with the scalar reference on argmax for
 *      every sample (prediction-level equivalence).
 *   2. All eligible kernels stay within AI_KERNEL_GATE_DIFF of the scalar
 *      reference at logit level on every sample (value-level equivalence).
 *
 * These run against the SAME exported model.bin/samples.bin the unikernel
 * boots with, so a drift between environments is caught before QEMU.
 *
 * A CPU without AVX2/FMA skips its SIMD assertions explicitly (printing
 * "SKIP"); the test never passes vacuously without saying so.
 */

static void *read_all(const char *path, size_t *size_out)
{
    FILE *file = fopen(path, "rb");
    long length;
    void *buffer;

    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    buffer = malloc((size_t)length);
    if (buffer == NULL ||
        fread(buffer, 1, (size_t)length, file) != (size_t)length) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size_out = (size_t)length;
    return buffer;
}

static int check_kernel(const char *name, ai_kernel_id_t kernel,
                        const ai_model_t *model, const ai_sample_t *samples,
                        uint32_t count, uint32_t *worst_diff)
{
    float reference[AI_OUTPUTS];
    float candidate[AI_OUTPUTS];
    uint32_t mismatches = 0;
    uint32_t i;

    *worst_diff = 0;
    for (i = 0; i < count; ++i) {
        uint32_t diff;

        ai_infer_kernel(model, samples[i].pixels, reference,
                        AI_KERNEL_SCALAR);
        ai_infer_kernel(model, samples[i].pixels, candidate, kernel);
        diff = ai_output_diff(reference, candidate);
        if (diff > *worst_diff) {
            *worst_diff = diff;
        }
        if (ai_argmax(reference) != ai_argmax(candidate)) {
            ++mismatches;
        }
    }
    printf("  %-8s worst diff %8u / %u  argmax mismatches %u / %u\n",
           name, *worst_diff, AI_KERNEL_GATE_DIFF, mismatches, count);
    return mismatches == 0 && *worst_diff <= AI_KERNEL_GATE_DIFF;
}

int main(int argc, char **argv)
{
    ai_caps_t caps;
    size_t model_bytes = 0;
    size_t samples_bytes = 0;
    ai_model_t *model;
    void *samples_raw;
    const ai_sample_t *samples;
    uint32_t sample_count = 0;
    uint32_t worst = 0;
    int failures = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.bin> <samples.bin>\n", argv[0]);
        return 2;
    }
    model = (ai_model_t *)read_all(argv[1], &model_bytes);
    samples_raw = read_all(argv[2], &samples_bytes);
    if (model == NULL || samples_raw == NULL ||
        model_bytes != sizeof(ai_model_t) ||
        !ai_model_valid(model, model_bytes) ||
        !ai_samples_valid(samples_raw, samples_bytes, &sample_count)) {
        fputs("error: unable to load model/samples\n", stderr);
        free(model);
        free(samples_raw);
        return 1;
    }
    samples = (const ai_sample_t *)((const uint8_t *)samples_raw +
                                    sizeof(ai_sample_header_t));

    ai_probe_caps(&caps);
    printf("cpu: avx=%d avx2=%d fma=%d state=%d\n",
           caps.has_avx, caps.has_avx2, caps.has_fma, caps.avx_state_ready);

    if (!check_kernel("scalar", AI_KERNEL_SCALAR, model, samples,
                      sample_count, &worst)) {
        ++failures;   /* scalar must self-agree exactly (diff 0) */
    }

    if (ai_kernel_eligible(AI_KERNEL_AVX2, &caps)) {
        if (!check_kernel("avx2", AI_KERNEL_AVX2, model, samples,
                          sample_count, &worst)) {
            ++failures;
        }
    } else {
        printf("  avx2     SKIP (not eligible on this CPU)\n");
    }
    if (ai_kernel_eligible(AI_KERNEL_FMA, &caps)) {
        if (!check_kernel("fma", AI_KERNEL_FMA, model, samples,
                          sample_count, &worst)) {
            ++failures;
        }
    } else {
        printf("  fma      SKIP (not eligible on this CPU)\n");
    }

    free(model);
    free(samples_raw);
    if (failures != 0) {
        fputs("dispatch equivalence FAILED\n", stderr);
        return 1;
    }
    puts("dispatch equivalence passed");
    return 0;
}
