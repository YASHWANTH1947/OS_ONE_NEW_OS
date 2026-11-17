#include "ai_service.h"

#include <float.h>

static const char ai_service_vocabulary[TEXT_AI_VOCAB + 1u] =
    " abcdefghijklmnopqrstuvwxyz.,!?0123456789\n";

static uint64_t ai_service_read_tsc(void)
{
    uint32_t low;
    uint32_t high;

    __asm__ volatile ("lfence; rdtsc; lfence"
                      : "=a"(low), "=d"(high) : : "memory");
    return ((uint64_t)high << 32) | low;
}

static size_t ai_service_align(size_t value)
{
    return (value + AI_SERVICE_ARENA_ALIGNMENT - 1u) &
           ~(AI_SERVICE_ARENA_ALIGNMENT - 1u);
}

static uint32_t ai_service_index(char value)
{
    uint32_t index;

    for (index = 0; index < TEXT_AI_VOCAB; ++index) {
        if (ai_service_vocabulary[index] == value) {
            return index;
        }
    }
    return 0u;
}

static char ai_service_char(uint32_t index)
{
    return index < TEXT_AI_VOCAB ? ai_service_vocabulary[index] : '?';
}

static float ai_service_tanh(float value)
{
    float squared;

    if (value > 3.0f) {
        return 1.0f;
    }
    if (value < -3.0f) {
        return -1.0f;
    }
    squared = value * value;
    return value * (27.0f + squared) / (27.0f + 9.0f * squared);
}

static ai_service_status_t ai_service_validate_model(const void *model,
                                                     size_t model_bytes)
{
    const text_ai_model_t *text_model =
        (const text_ai_model_t *)model;

    if (model == 0) {
        return AI_SERVICE_ERR_MODEL_NULL;
    }
    if (model_bytes != sizeof(*text_model)) {
        return AI_SERVICE_ERR_MODEL_SIZE;
    }
    if (text_model->version != TEXT_AI_VERSION) {
        return AI_SERVICE_ERR_MODEL_VERSION;
    }
    if (text_model->magic != TEXT_AI_MAGIC ||
        text_model->vocab != TEXT_AI_VOCAB ||
        text_model->hidden != TEXT_AI_HIDDEN ||
        text_model->reserved != 0u) {
        return AI_SERVICE_ERR_MODEL_FORMAT;
    }
    return AI_SERVICE_OK;
}

static void ai_service_reset_metrics(ai_service_t *service)
{
    service->metrics.arena_capacity = service->arena_capacity;
    service->metrics.arena_used = service->arena_used;
    service->metrics.arena_peak = service->arena_peak;
}

ai_service_status_t ai_init_arena(ai_service_t *service, void *arena,
                                  size_t arena_bytes)
{
    size_t cursor;
    size_t required;

    if (service == 0) {
        return AI_SERVICE_ERR_ARGUMENT;
    }
    if (arena == 0) {
        return AI_SERVICE_ERR_ARENA_NULL;
    }
    if (arena_bytes < AI_SERVICE_ARENA_BYTES) {
        return AI_SERVICE_ERR_ARENA_SIZE;
    }

    cursor = 0;
    cursor = ai_service_align(cursor);
    cursor += AI_SERVICE_PROMPT_CAPACITY;
    cursor = ai_service_align(cursor);
    cursor += AI_SERVICE_OUTPUT_CAPACITY;
    cursor = ai_service_align(cursor);
    cursor += TEXT_AI_HIDDEN * sizeof(float);
    cursor = ai_service_align(cursor);
    cursor += TEXT_AI_HIDDEN * sizeof(float);
    required = cursor;
    if (required > arena_bytes) {
        return AI_SERVICE_ERR_ARENA_EXHAUSTED;
    }

    service->model = 0;
    service->arena = (uint8_t *)arena;
    service->arena_capacity = arena_bytes;
    service->arena_used = required;
    service->arena_peak = required;
    service->prompt_buffer = (char *)(void *)service->arena;
    cursor = ai_service_align((size_t)AI_SERVICE_PROMPT_CAPACITY);
    service->output_buffer = (char *)(void *)(service->arena + cursor);
    cursor = ai_service_align(cursor + AI_SERVICE_OUTPUT_CAPACITY);
    service->hidden = (float *)(void *)(service->arena + cursor);
    cursor = ai_service_align(cursor + TEXT_AI_HIDDEN * sizeof(float));
    service->next_hidden = (float *)(void *)(service->arena + cursor);
    service->metrics = (ai_metrics_t){0};
    ai_service_reset_metrics(service);
    return AI_SERVICE_OK;
}

ai_service_status_t ai_load_model(ai_service_t *service, const void *model,
                                  size_t model_bytes)
{
    ai_service_status_t status;

    if (service == 0) {
        return AI_SERVICE_ERR_ARGUMENT;
    }
    if (service->arena == 0) {
        return AI_SERVICE_ERR_NOT_INITIALIZED;
    }
    if (service->model != 0) {
        return AI_SERVICE_ERR_ALREADY_LOADED;
    }
    status = ai_service_validate_model(model, model_bytes);
    if (status != AI_SERVICE_OK) {
        return status;
    }
    service->model = (const text_ai_model_t *)model;
    service->metrics.model_bytes = model_bytes;
    return AI_SERVICE_OK;
}

ai_service_status_t ai_generate_stream(ai_service_t *service,
                                       const char *prompt, size_t prompt_bytes,
                                       const ai_generate_config_t *config,
                                       ai_token_callback_t callback,
                                       void *callback_context)
{
    const text_ai_model_t *model;
    uint64_t start_cycles;
    uint64_t elapsed_cycles;
    size_t generated = 0;
    uint32_t step;

    if (service == 0 || config == 0) {
        return AI_SERVICE_ERR_ARGUMENT;
    }
    if (service->arena == 0) {
        return AI_SERVICE_ERR_NOT_INITIALIZED;
    }
    if (service->model == 0) {
        return AI_SERVICE_ERR_NOT_INITIALIZED;
    }
    if (callback == 0) {
        return AI_SERVICE_ERR_CALLBACK;
    }
    if (prompt == 0 && prompt_bytes != 0) {
        return AI_SERVICE_ERR_ARGUMENT;
    }
    if (prompt_bytes > TEXT_AI_PROMPT_LIMIT) {
        return AI_SERVICE_ERR_PROMPT_TOO_LONG;
    }
    if (config->max_tokens == 0 ||
        config->max_tokens > TEXT_AI_OUTPUT_LIMIT) {
        return AI_SERVICE_ERR_GENERATION_LIMIT;
    }
    if (config->output == 0 ||
        config->output_capacity < (size_t)config->max_tokens + 1u) {
        return AI_SERVICE_ERR_OUTPUT_TOO_SMALL;
    }

    for (step = 0; step < (uint32_t)prompt_bytes; ++step) {
        service->prompt_buffer[step] = prompt[step];
    }
    service->prompt_buffer[prompt_bytes] = '\0';
    for (step = 0; step < TEXT_AI_HIDDEN; ++step) {
        service->hidden[step] = 0.0f;
    }
    model = service->model;
    start_cycles = ai_service_read_tsc();
    for (step = 0; step < (uint32_t)prompt_bytes; ++step) {
        uint32_t token = ai_service_index(service->prompt_buffer[step]);
        uint32_t hidden_index;

        for (hidden_index = 0; hidden_index < TEXT_AI_HIDDEN;
             ++hidden_index) {
            service->hidden[hidden_index] = ai_service_tanh(
                model->input[token * TEXT_AI_HIDDEN + hidden_index] +
                model->hidden_bias[hidden_index]);
        }
    }

    while (generated < config->max_tokens) {
        uint32_t best = 0;
        float best_score = -FLT_MAX;
        uint32_t token;

        for (token = 0; token < TEXT_AI_VOCAB; ++token) {
            float score = model->output_bias[token];
            uint32_t hidden_index;

            for (hidden_index = 0; hidden_index < TEXT_AI_HIDDEN;
                 ++hidden_index) {
                score += service->hidden[hidden_index] *
                    model->output[hidden_index * TEXT_AI_VOCAB + token];
            }
            if (score > best_score) {
                best_score = score;
                best = token;
            }
        }

        service->output_buffer[generated] = ai_service_char(best);
        config->output[generated] = service->output_buffer[generated];
        if (callback(&config->output[generated], 1u, (uint32_t)generated,
                     callback_context) != 0) {
            elapsed_cycles = ai_service_read_tsc() - start_cycles;
            service->metrics.last_generation_cycles = elapsed_cycles;
            service->metrics.total_generation_cycles += elapsed_cycles;
            service->metrics.generations += 1u;
            service->metrics.tokens_generated += (uint32_t)generated;
            return AI_SERVICE_ERR_CALLBACK;
        }
        ++generated;
        if (best == ai_service_index('\n')) {
            break;
        }
        {
            uint32_t hidden_index;

            for (hidden_index = 0; hidden_index < TEXT_AI_HIDDEN;
                 ++hidden_index) {
                float value = model->hidden_bias[hidden_index];
                uint32_t recurrent_index;

                for (recurrent_index = 0;
                     recurrent_index < TEXT_AI_HIDDEN; ++recurrent_index) {
                    value += service->hidden[recurrent_index] *
                        model->recurrent[recurrent_index * TEXT_AI_HIDDEN +
                                         hidden_index];
                }
                service->next_hidden[hidden_index] =
                    ai_service_tanh(value);
            }
            for (hidden_index = 0; hidden_index < TEXT_AI_HIDDEN;
                 ++hidden_index) {
                service->hidden[hidden_index] =
                    service->next_hidden[hidden_index];
            }
        }
    }
    config->output[generated] = '\0';
    elapsed_cycles = ai_service_read_tsc() - start_cycles;
    service->metrics.last_generation_cycles = elapsed_cycles;
    service->metrics.total_generation_cycles += elapsed_cycles;
    service->metrics.generations += 1u;
    service->metrics.tokens_generated += (uint32_t)generated;
    service->metrics.arena_used = service->arena_used;
    service->metrics.arena_peak = service->arena_peak;
    return AI_SERVICE_OK;
}

ai_service_status_t ai_get_metrics(const ai_service_t *service,
                                   ai_metrics_t *metrics)
{
    if (service == 0 || metrics == 0) {
        return AI_SERVICE_ERR_ARGUMENT;
    }
    if (service->arena == 0) {
        return AI_SERVICE_ERR_NOT_INITIALIZED;
    }
    *metrics = service->metrics;
    return AI_SERVICE_OK;
}

const char *ai_service_status_string(ai_service_status_t status)
{
    switch (status) {
    case AI_SERVICE_OK:
        return "ok";
    case AI_SERVICE_ERR_ARGUMENT:
        return "invalid argument";
    case AI_SERVICE_ERR_NOT_INITIALIZED:
        return "service not initialized";
    case AI_SERVICE_ERR_ALREADY_LOADED:
        return "model already loaded";
    case AI_SERVICE_ERR_MODEL_NULL:
        return "model is null";
    case AI_SERVICE_ERR_MODEL_SIZE:
        return "invalid model size";
    case AI_SERVICE_ERR_MODEL_FORMAT:
        return "invalid model format";
    case AI_SERVICE_ERR_MODEL_VERSION:
        return "unsupported model version";
    case AI_SERVICE_ERR_ARENA_NULL:
        return "arena is null";
    case AI_SERVICE_ERR_ARENA_SIZE:
        return "arena is too small";
    case AI_SERVICE_ERR_ARENA_EXHAUSTED:
        return "arena layout does not fit";
    case AI_SERVICE_ERR_PROMPT_TOO_LONG:
        return "prompt is too long";
    case AI_SERVICE_ERR_OUTPUT_TOO_SMALL:
        return "output buffer is too small";
    case AI_SERVICE_ERR_GENERATION_LIMIT:
        return "generation limit is invalid";
    case AI_SERVICE_ERR_CALLBACK:
        return "token callback failed";
    default:
        return "unknown service error";
    }
}
