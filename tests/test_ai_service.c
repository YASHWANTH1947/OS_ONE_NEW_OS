#include "ai_service.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint32_t calls;
    uint32_t bytes;
    char text[AI_SERVICE_OUTPUT_CAPACITY];
} callback_state_t;

static int collect_token(const char *text, size_t bytes,
                         uint32_t token_index, void *context)
{
    callback_state_t *state = (callback_state_t *)context;
    size_t index;

    assert(token_index < AI_SERVICE_OUTPUT_CAPACITY);
    for (index = 0; index < bytes; ++index) {
        assert((size_t)token_index + index < sizeof(state->text) - 1u);
        state->text[token_index + index] = text[index];
    }
    state->calls += 1u;
    state->bytes += (uint32_t)bytes;
    state->text[state->bytes] = '\0';
    return 0;
}

static void *read_model(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    long length;
    void *buffer;

    assert(file != NULL);
    assert(fseek(file, 0, SEEK_END) == 0);
    length = ftell(file);
    assert(length == (long)sizeof(text_ai_model_t));
    assert(fseek(file, 0, SEEK_SET) == 0);
    buffer = malloc((size_t)length);
    assert(buffer != NULL);
    assert(fread(buffer, 1, (size_t)length, file) == (size_t)length);
    assert(fclose(file) == 0);
    *bytes = (size_t)length;
    return buffer;
}

static void test_validation(const text_ai_model_t *model, size_t model_bytes)
{
    ai_service_t service = {0};
    uint8_t arena[AI_SERVICE_ARENA_BYTES]
        __attribute__((aligned(AI_SERVICE_ARENA_ALIGNMENT)));

    assert(ai_init_arena(NULL, arena, sizeof(arena)) ==
           AI_SERVICE_ERR_ARGUMENT);
    assert(ai_init_arena(&service, NULL, sizeof(arena)) ==
           AI_SERVICE_ERR_ARENA_NULL);
    assert(ai_init_arena(&service, arena, sizeof(arena) - 1u) ==
           AI_SERVICE_ERR_ARENA_SIZE);
    assert(ai_init_arena(&service, arena, sizeof(arena)) == AI_SERVICE_OK);
    assert(ai_load_model(&service, NULL, model_bytes) ==
           AI_SERVICE_ERR_MODEL_NULL);
    assert(ai_load_model(&service, model, model_bytes - 1u) ==
           AI_SERVICE_ERR_MODEL_SIZE);
    assert(ai_load_model(&service, model, model_bytes) == AI_SERVICE_OK);
    assert(ai_load_model(&service, model, model_bytes) ==
           AI_SERVICE_ERR_ALREADY_LOADED);
    assert(service.model == model);
}

static void test_generation(const text_ai_model_t *model, size_t model_bytes)
{
    ai_service_t service = {0};
    ai_generate_config_t config;
    ai_metrics_t metrics;
    callback_state_t callback = {0};
    uint8_t arena[AI_SERVICE_ARENA_BYTES]
        __attribute__((aligned(AI_SERVICE_ARENA_ALIGNMENT)));
    char output[AI_SERVICE_OUTPUT_CAPACITY];
    const char prompt[] = "status";

    assert(ai_init_arena(&service, arena, sizeof(arena)) == AI_SERVICE_OK);
    assert(ai_load_model(&service, model, model_bytes) == AI_SERVICE_OK);
    assert(ai_get_metrics(&service, &metrics) == AI_SERVICE_OK);
    assert(metrics.model_bytes == model_bytes);
    assert(metrics.arena_used > 0u);

    config.max_tokens = 32u;
    config.output = output;
    config.output_capacity = sizeof(output);
    assert(ai_generate_stream(&service, prompt, sizeof(prompt) - 1u,
                              &config, collect_token, &callback) ==
           AI_SERVICE_OK);
    assert(callback.calls > 0u);
    assert(callback.bytes == callback.calls);
    assert(output[callback.bytes] == '\0');
    assert(ai_get_metrics(&service, &metrics) == AI_SERVICE_OK);
    assert(metrics.generations == 1u);
    assert(metrics.tokens_generated == callback.bytes);
    assert(metrics.last_generation_cycles > 0u);

    assert(ai_generate_stream(&service, prompt, TEXT_AI_PROMPT_LIMIT + 1u,
                              &config, collect_token, &callback) ==
           AI_SERVICE_ERR_PROMPT_TOO_LONG);
    config.output_capacity = 1u;
    assert(ai_generate_stream(&service, prompt, sizeof(prompt) - 1u,
                              &config, collect_token, &callback) ==
           AI_SERVICE_ERR_OUTPUT_TOO_SMALL);
    config.output_capacity = sizeof(output);
    config.max_tokens = 0u;
    assert(ai_generate_stream(&service, prompt, sizeof(prompt) - 1u,
                              &config, collect_token, &callback) ==
           AI_SERVICE_ERR_GENERATION_LIMIT);
    config.max_tokens = 32u;
    assert(ai_generate_stream(&service, prompt, sizeof(prompt) - 1u,
                              &config, NULL, &callback) ==
           AI_SERVICE_ERR_CALLBACK);
}

int main(int argc, char **argv)
{
    size_t model_bytes;
    void *raw_model;

    assert(argc == 2);
    raw_model = read_model(argv[1], &model_bytes);
    test_validation((const text_ai_model_t *)raw_model, model_bytes);
    test_generation((const text_ai_model_t *)raw_model, model_bytes);
    free(raw_model);
    puts("AI service tests passed.");
    return 0;
}
