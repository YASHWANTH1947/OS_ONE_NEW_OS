#ifndef AI_SERVICE_H
#define AI_SERVICE_H

#include <stddef.h>
#include <stdint.h>

#include "text_ai.h"

#define AI_SERVICE_ARENA_BYTES 16384u
#define AI_SERVICE_ARENA_ALIGNMENT 64u
#define AI_SERVICE_PROMPT_CAPACITY (TEXT_AI_PROMPT_LIMIT + 1u)
#define AI_SERVICE_OUTPUT_CAPACITY (TEXT_AI_OUTPUT_LIMIT + 1u)

typedef enum {
    AI_SERVICE_OK = 0,
    AI_SERVICE_ERR_ARGUMENT,
    AI_SERVICE_ERR_NOT_INITIALIZED,
    AI_SERVICE_ERR_ALREADY_LOADED,
    AI_SERVICE_ERR_MODEL_NULL,
    AI_SERVICE_ERR_MODEL_SIZE,
    AI_SERVICE_ERR_MODEL_FORMAT,
    AI_SERVICE_ERR_MODEL_VERSION,
    AI_SERVICE_ERR_ARENA_NULL,
    AI_SERVICE_ERR_ARENA_SIZE,
    AI_SERVICE_ERR_ARENA_EXHAUSTED,
    AI_SERVICE_ERR_PROMPT_TOO_LONG,
    AI_SERVICE_ERR_OUTPUT_TOO_SMALL,
    AI_SERVICE_ERR_GENERATION_LIMIT,
    AI_SERVICE_ERR_CALLBACK
} ai_service_status_t;

typedef int (*ai_token_callback_t)(const char *text, size_t bytes,
                                   uint32_t token_index, void *context);

typedef struct {
    uint32_t max_tokens;
    char *output;
    size_t output_capacity;
} ai_generate_config_t;

typedef struct {
    size_t arena_capacity;
    size_t arena_used;
    size_t arena_peak;
    size_t model_bytes;
    uint32_t generations;
    uint32_t tokens_generated;
    uint64_t last_generation_cycles;
    uint64_t total_generation_cycles;
} ai_metrics_t;

typedef struct {
    const text_ai_model_t *model;
    uint8_t *arena;
    size_t arena_capacity;
    size_t arena_used;
    size_t arena_peak;
    char *prompt_buffer;
    char *output_buffer;
    float *hidden;
    float *next_hidden;
    ai_metrics_t metrics;
} ai_service_t;

ai_service_status_t ai_init_arena(ai_service_t *service, void *arena,
                                  size_t arena_bytes);

ai_service_status_t ai_load_model(ai_service_t *service, const void *model,
                                  size_t model_bytes);

ai_service_status_t ai_generate_stream(ai_service_t *service,
                                       const char *prompt, size_t prompt_bytes,
                                       const ai_generate_config_t *config,
                                       ai_token_callback_t callback,
                                       void *callback_context);

ai_service_status_t ai_get_metrics(const ai_service_t *service,
                                   ai_metrics_t *metrics);

const char *ai_service_status_string(ai_service_status_t status);

#endif
