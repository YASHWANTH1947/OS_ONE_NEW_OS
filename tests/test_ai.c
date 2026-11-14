#include "ai.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Host-side checks for the inference core and for the exported artifacts. These
 * read the same model.bin and samples.bin the kernel boots with, so a broken
 * export fails here before it ever reaches QEMU.
 */

static void *read_file(const char *path, size_t bytes)
{
    void *buffer = malloc(bytes);
    FILE *file = fopen(path, "rb");

    assert(file != NULL);
    assert(buffer != NULL);
    assert(fread(buffer, 1, bytes, file) == bytes);
    assert(fgetc(file) == EOF);
    assert(!ferror(file));
    assert(fclose(file) == 0);
    return buffer;
}

static void test_model_validation(void)
{
    ai_model_t *model = (ai_model_t *)calloc(1, sizeof(*model));

    assert(model != NULL);
    assert(sizeof(*model) == 407100u);

    model->magic = AI_MAGIC;
    model->version = AI_VERSION;
    model->inputs = AI_INPUTS;
    model->hidden = AI_HIDDEN;
    model->outputs = AI_OUTPUTS;

    assert(ai_model_valid(model, sizeof(*model)));
    assert(!ai_model_valid(model, sizeof(*model) - 1u));
    assert(!ai_model_valid(NULL, sizeof(*model)));

    model->version += 1u;
    assert(!ai_model_valid(model, sizeof(*model)));

    free(model);
}

static void test_relu_and_argmax(void)
{
    ai_model_t *model = (ai_model_t *)calloc(1, sizeof(*model));
    float input[AI_INPUTS] = {0};
    float output[AI_OUTPUTS];

    assert(model != NULL);
    model->b1[0] = -1.0f;
    model->w1[0] = 2.0f;
    model->b1[1] = 2.0f;
    model->w1[AI_INPUTS + 1u] = -2.0f;
    model->w2[4u * AI_HIDDEN] = 1.0f;
    model->b2[5] = 1.0f;
    input[0] = 1.0f;
    input[1] = 2.0f;

    ai_infer(model, input, output);
    assert(output[4] == 1.0f);
    assert(output[5] == 1.0f);
    assert(ai_argmax(output) == 4u);

    free(model);
}

static void test_sample_validation(void)
{
    union {
        uint64_t alignment;
        uint8_t bytes[sizeof(ai_sample_header_t) + sizeof(ai_sample_t)];
    } storage = {0};
    uint8_t *bytes = storage.bytes;
    ai_sample_header_t *header = (ai_sample_header_t *)bytes;
    ai_sample_t *sample =
        (ai_sample_t *)(bytes + sizeof(ai_sample_header_t));
    uint32_t count = 0;
    union {
        uint32_t bits;
        float value;
    } nan_value = {0x7FC00000u};

    header->magic = AI_SAMPLE_MAGIC;
    header->version = AI_SAMPLE_VERSION;
    header->count = 1u;
    assert(ai_samples_valid(bytes, sizeof(storage.bytes), &count));
    assert(count == 1u);
    assert(!ai_samples_valid(NULL, sizeof(storage.bytes), NULL));
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes) - 1u, NULL));
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes) + 1u, NULL));

    header->reserved = 1u;
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes), NULL));
    header->reserved = 0u;

    sample->label = AI_OUTPUTS;
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes), NULL));
    sample->label = 0u;

    sample->pixels[0] = -0.01f;
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes), NULL));
    sample->pixels[0] = 1.01f;
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes), NULL));
    sample->pixels[0] = nan_value.value;
    assert(!ai_samples_valid(bytes, sizeof(storage.bytes), NULL));
}

static void *read_samples(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    long length;
    void *buffer;

    assert(file != NULL);
    assert(fseek(file, 0, SEEK_END) == 0);
    length = ftell(file);
    assert(length > 0);
    assert(fseek(file, 0, SEEK_SET) == 0);

    buffer = malloc((size_t)length);
    assert(buffer != NULL);
    assert(fread(buffer, 1, (size_t)length, file) == (size_t)length);
    assert(fgetc(file) == EOF);
    assert(!ferror(file));
    assert(fclose(file) == 0);
    *bytes = (size_t)length;
    return buffer;
}

static void test_artifacts(const char *model_path, const char *sample_path)
{
    ai_model_t *model = (ai_model_t *)read_file(model_path, sizeof(ai_model_t));
    const ai_sample_t *samples;
    float first[AI_OUTPUTS];
    float repeat[AI_OUTPUTS];
    uint32_t count = 0;
    uint32_t correct = 0;
    uint32_t index;
    void *raw;
    size_t bytes;

    assert(ai_model_valid(model, sizeof(ai_model_t)));
    raw = read_samples(sample_path, &bytes);
    assert(ai_samples_valid(raw, bytes, &count));
    assert(count > 0);
    samples = (const ai_sample_t *)((const uint8_t *)raw +
                                    sizeof(ai_sample_header_t));

    /* Deterministic: the same input must always produce the same logits. */
    ai_infer(model, samples[0].pixels, first);
    ai_infer(model, samples[0].pixels, repeat);
    assert(memcmp(first, repeat, sizeof(first)) == 0);

    for (index = 0; index < count; ++index) {
        ai_infer(model, samples[index].pixels, first);
        if (ai_argmax(first) == samples[index].label) {
            ++correct;
        }
    }

    printf("  model ............ %s\n", model_path);
    printf("  samples .......... %u\n", count);
    printf("  correct .......... %u / %u (%.2f%%)\n", correct, count,
           100.0 * (double)correct / (double)count);

    free(raw);
    free(model);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.bin> <samples.bin>\n", argv[0]);
        return 2;
    }

    test_model_validation();
    test_relu_and_argmax();
    test_sample_validation();
    test_artifacts(argv[1], argv[2]);
    puts("All inference and artifact tests passed.");
    return 0;
}
