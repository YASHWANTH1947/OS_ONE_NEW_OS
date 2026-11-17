#include "llm_runtime.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t header_bytes;
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t layers;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t vocab_size;
    uint32_t seq_flags;
    uint32_t checksum;
} model_header_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab_size;
    uint32_t max_token_length;
    uint32_t records_bytes;
    uint32_t checksum;
} tokenizer_header_t;

static int receive_token(const uint8_t *bytes, size_t length,
                         uint32_t token_index, void *context)
{
    uint32_t *received = (uint32_t *)context;

    assert(length == 1u);
    assert(bytes[0] == 'a' || bytes[0] == 'b');
    assert(token_index < LLM_OUTPUT_LIMIT);
    *received += 1u;
    return 0;
}

int main(void)
{
    union {
        uint64_t alignment;
        uint8_t bytes[sizeof(model_header_t) + 164u * sizeof(float)];
    } model_storage = {0};
    union {
        uint64_t alignment;
        uint8_t bytes[sizeof(tokenizer_header_t) + 18u];
    } tokenizer_storage = {0};
    static uint8_t arena[LLM_RUNTIME_ARENA_BYTES]
        __attribute__((aligned(64)));
    model_header_t *model = (model_header_t *)model_storage.bytes;
    tokenizer_header_t *tokenizer =
        (tokenizer_header_t *)tokenizer_storage.bytes;
    float *weights = (float *)(void *)(model_storage.bytes +
                                       sizeof(model_header_t));
    uint8_t *records = tokenizer_storage.bytes + sizeof(*tokenizer);
    llm_runtime_t runtime = {0};
    llm_metrics_t metrics;
    uint32_t received = 0;
    llm_status_t status;
    uint32_t model_checksum = 2166136261u;

    model->magic = LLM_MODEL_MAGIC;
    model->version = LLM_FORMAT_VERSION;
    model->header_bytes = sizeof(*model);
    model->dim = 4u;
    model->hidden_dim = 4u;
    model->layers = 1u;
    model->heads = 1u;
    model->kv_heads = 1u;
    model->vocab_size = 2u;
    model->seq_flags = 8u | (1u << 16);
    for (uint32_t index = 0; index < 8u; ++index) {
        weights[index] = index < 4u ? 0.1f : 0.2f;
    }
    for (uint32_t index = 8u; index < 164u; ++index) {
        weights[index] = 0.0f;
    }
    for (uint32_t index = 8u; index < 12u; ++index) {
        weights[index] = 1.0f;
    }
    for (uint32_t index = 76u; index < 80u; ++index) {
        weights[index] = 1.0f;
    }
    for (uint32_t index = 128u; index < 132u; ++index) {
        weights[index] = 1.0f;
    }
    model->checksum = 0u;
    for (uint32_t index = 0; index < 164u * sizeof(float); ++index) {
        model_checksum = (model_checksum ^
                          ((uint8_t *)weights)[index]) * 16777619u;
    }
    model->checksum = model_checksum;

    tokenizer->magic = LLM_TOKENIZER_MAGIC;
    tokenizer->version = LLM_FORMAT_VERSION;
    tokenizer->header_bytes = sizeof(*tokenizer);
    tokenizer->vocab_size = 2u;
    tokenizer->max_token_length = 1u;
    tokenizer->records_bytes = 18u;
    *(float *)(void *)records = 0.0f;
    *(uint32_t *)(void *)(records + 4u) = 1u;
    records[8] = 'a';
    *(float *)(void *)(records + 9u) = 0.0f;
    *(uint32_t *)(void *)(records + 13u) = 1u;
    records[17] = 'b';
    tokenizer->checksum = 2166136261u;
    for (uint32_t index = 0; index < 18u; ++index) {
        tokenizer->checksum =
            (tokenizer->checksum ^ records[index]) * 16777619u;
    }

    status = llm_runtime_init(&runtime, arena, sizeof(arena),
                              model_storage.bytes, sizeof(model_storage.bytes),
                              tokenizer_storage.bytes,
                              sizeof(tokenizer_storage.bytes));
    assert(status == LLM_OK);
    status = llm_generate(&runtime, "a", 1u, 2u, receive_token, &received);
    assert(status == LLM_OK);
    assert(received == 2u);
    assert(llm_get_metrics(&runtime, &metrics) == LLM_OK);
    assert(metrics.prompt_tokens == 1u);
    assert(metrics.generated_tokens == 2u);
    assert(metrics.arena_used > 0u);
    puts("LLM runtime tests passed.");
    return 0;
}
