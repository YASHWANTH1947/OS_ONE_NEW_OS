#ifndef LLM_RUNTIME_H
#define LLM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define LLM_MODEL_MAGIC 0x314D4C4Cu
#define LLM_TOKENIZER_MAGIC 0x315A544Bu
#define LLM_FORMAT_VERSION 1u
#define LLM_CONTEXT_LIMIT 128u
#define LLM_OUTPUT_LIMIT 128u
#define LLM_RUNTIME_ARENA_BYTES (2u * 1024u * 1024u)

typedef enum {
    LLM_OK = 0,
    LLM_ERR_ARGUMENT,
    LLM_ERR_MODEL_SIZE,
    LLM_ERR_MODEL_FORMAT,
    LLM_ERR_MODEL_VERSION,
    LLM_ERR_TOKENIZER_SIZE,
    LLM_ERR_TOKENIZER_FORMAT,
    LLM_ERR_TOKENIZER_VERSION,
    LLM_ERR_ARENA_NULL,
    LLM_ERR_ARENA_SIZE,
    LLM_ERR_ARENA_EXHAUSTED,
    LLM_ERR_PROMPT_TOO_LONG,
    LLM_ERR_OUTPUT_LIMIT,
    LLM_ERR_CALLBACK,
    LLM_ERR_NOT_INITIALIZED
} llm_status_t;

typedef int (*llm_token_callback_t)(const uint8_t *bytes, size_t length,
                                    uint32_t token_index, void *context);

typedef struct {
    size_t arena_capacity;
    size_t arena_used;
    size_t arena_peak;
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint64_t validation_cycles;
    uint64_t tokenization_cycles;
    uint64_t prefill_cycles;
    uint64_t generation_cycles;
    uint64_t last_token_cycles;
} llm_metrics_t;

typedef struct {
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t layers;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t shared_classifier;
    uint32_t kv_dim;
    const float *embedding;
    const float *rms_att;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *rms_ffn;
    const float *w1;
    const float *w2;
    const float *w3;
    const float *rms_final;
    const float *classifier;
} llm_model_t;

typedef struct {
    uint32_t vocab_size;
    uint32_t max_token_length;
    const uint8_t **token_data;
    uint32_t *token_lengths;
} llm_tokenizer_t;

typedef struct {
    llm_model_t model;
    llm_tokenizer_t tokenizer;
    uint8_t *arena;
    size_t arena_capacity;
    size_t arena_used;
    size_t arena_peak;
    uint32_t *token_ids;
    float *x;
    float *next;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *att;
    float *ffn_hidden;
    float *ffn_gate;
    float *logits;
    float *key_cache;
    float *value_cache;
    llm_metrics_t metrics;
} llm_runtime_t;

llm_status_t llm_runtime_init(llm_runtime_t *runtime, void *arena,
                              size_t arena_bytes, const void *model,
                              size_t model_bytes, const void *tokenizer,
                              size_t tokenizer_bytes);

llm_status_t llm_generate(llm_runtime_t *runtime, const char *prompt,
                          size_t prompt_bytes, uint32_t max_tokens,
                          llm_token_callback_t callback, void *context);

llm_status_t llm_get_metrics(const llm_runtime_t *runtime,
                             llm_metrics_t *metrics);

const char *llm_status_string(llm_status_t status);

#endif
