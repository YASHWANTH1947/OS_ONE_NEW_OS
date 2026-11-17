#include "llm_runtime.h"

#define LLM_EPSILON 0.000001f
#define LLM_PI 3.14159265358979323846f
#define LLM_TWO_PI (2.0f * LLM_PI)

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
} llm_model_header_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab_size;
    uint32_t max_token_length;
    uint32_t records_bytes;
    uint32_t checksum;
} llm_tokenizer_header_t;

static uint64_t llm_read_tsc(void)
{
    uint32_t low;
    uint32_t high;

    __asm__ volatile ("lfence; rdtsc; lfence"
                      : "=a"(low), "=d"(high) : : "memory");
    return ((uint64_t)high << 32) | low;
}

static size_t llm_align(size_t value)
{
    return (value + 63u) & ~63u;
}

static uint32_t llm_checksum(const uint8_t *bytes, size_t length)
{
    uint32_t value = 2166136261u;
    size_t index;

    for (index = 0; index < length; ++index) {
        value = (value ^ bytes[index]) * 16777619u;
    }
    return value;
}

static float llm_rsqrt(float value)
{
    float result;
    uint32_t bits;

    if (value <= 0.0f) {
        return 0.0f;
    }
    __builtin_memcpy(&bits, &value, sizeof(bits));
    bits = 0x5F3759DFu - (bits >> 1);
    __builtin_memcpy(&result, &bits, sizeof(result));
    result *= 1.5f - 0.5f * value * result * result;
    result *= 1.5f - 0.5f * value * result * result;
    return result;
}

static float llm_wrap_angle(float value)
{
    while (value > LLM_PI) {
        value -= LLM_TWO_PI;
    }
    while (value < -LLM_PI) {
        value += LLM_TWO_PI;
    }
    return value;
}

static float llm_sin(float value)
{
    float x = llm_wrap_angle(value);
    float x2 = x * x;

    return x * (1.0f - x2 / 6.0f + (x2 * x2) / 120.0f -
                (x2 * x2 * x2) / 5040.0f);
}

static float llm_cos(float value)
{
    float x = llm_wrap_angle(value);
    float x2 = x * x;

    return 1.0f - x2 / 2.0f + (x2 * x2) / 24.0f -
           (x2 * x2 * x2) / 720.0f;
}

static float llm_rope_base(uint32_t head_size)
{
    float value = 0.0001f;
    uint32_t divisor = 2u;

    while (divisor < head_size) {
        value *= llm_rsqrt(value);
        divisor *= 2u;
    }
    return value;
}

static float llm_exp(float value)
{
    float term = 1.0f;
    float sum = 1.0f;
    uint32_t index;

    if (value < -10.0f) {
        return 0.0f;
    }
    if (value > 10.0f) {
        value = 10.0f;
    }
    for (index = 1; index <= 12u; ++index) {
        term *= value / (float)index;
        sum += term;
    }
    return sum > 0.0f ? sum : 0.0f;
}

static void llm_copy(float *destination, const float *source, uint32_t count)
{
    uint32_t index;

    for (index = 0; index < count; ++index) {
        destination[index] = source[index];
    }
}

static void llm_zero(float *values, uint32_t count)
{
    uint32_t index;

    for (index = 0; index < count; ++index) {
        values[index] = 0.0f;
    }
}

static void llm_matvec(float *output, const float *weights, const float *input,
                       uint32_t rows, uint32_t columns)
{
    uint32_t row;
    uint32_t column;

    for (row = 0; row < rows; ++row) {
        float sum = 0.0f;

        for (column = 0; column < columns; ++column) {
            sum += weights[row * columns + column] * input[column];
        }
        output[row] = sum;
    }
}

static void llm_rmsnorm(float *output, const float *input, const float *weight,
                        uint32_t count)
{
    float sum = 0.0f;
    float scale;
    uint32_t index;

    for (index = 0; index < count; ++index) {
        sum += input[index] * input[index];
    }
    scale = llm_rsqrt(sum / (float)count + LLM_EPSILON);
    for (index = 0; index < count; ++index) {
        output[index] = input[index] * scale * weight[index];
    }
}

static void llm_rope(float *values, uint32_t count, uint32_t position,
                     uint32_t head_size)
{
    uint32_t index;
    float base = llm_rope_base(head_size);

    for (index = 0; index < count; index += 2u) {
        uint32_t pair = index / 2u;
        float frequency = 1.0f;
        float angle;
        float cosine;
        float sine;
        float first;
        float second;

        for (uint32_t power = 0; power < pair; ++power) {
            frequency *= base;
        }
        angle = (float)position * frequency;
        cosine = llm_cos(angle);
        sine = llm_sin(angle);
        first = values[index];
        second = values[index + 1u];
        values[index] = first * cosine - second * sine;
        values[index + 1u] = first * sine + second * cosine;
    }
    (void)head_size;
}

static llm_status_t llm_attach_model(llm_model_t *model, const void *buffer,
                                      size_t bytes)
{
    const llm_model_header_t *header =
        (const llm_model_header_t *)buffer;
    const float *weights;
    size_t expected;
    uint32_t kv_dim;
    uint32_t head_size;
    uint32_t source_context;
    uint32_t flags;

    if (buffer == 0 || bytes < sizeof(*header)) {
        return LLM_ERR_MODEL_SIZE;
    }
    if (header->magic != LLM_MODEL_MAGIC ||
        header->header_bytes != sizeof(*header)) {
        return LLM_ERR_MODEL_FORMAT;
    }
    if (header->version != LLM_FORMAT_VERSION) {
        return LLM_ERR_MODEL_VERSION;
    }
    if (header->heads == 0 || header->kv_heads == 0 ||
        header->kv_heads > header->heads ||
        header->heads % header->kv_heads != 0 ||
        header->dim == 0 || header->dim % header->heads != 0 ||
        header->hidden_dim == 0 || header->layers == 0 ||
        header->vocab_size == 0 || header->vocab_size > 1000000u) {
        return LLM_ERR_MODEL_FORMAT;
    }
    kv_dim = header->dim * header->kv_heads / header->heads;
    head_size = header->dim / header->heads;
    source_context = header->seq_flags & 0xFFFFu;
    flags = header->seq_flags >> 16;
    if ((header->seq_flags & 0xFFFFu) == 0 ||
        (flags & ~1u) != 0u) {
        return LLM_ERR_MODEL_FORMAT;
    }
    expected = sizeof(*header) +
        ((size_t)header->vocab_size * header->dim +
         (size_t)header->layers * header->dim +
         (size_t)header->layers * header->dim * header->dim +
         (size_t)header->layers * header->dim * kv_dim * 2u +
         (size_t)header->layers * header->dim * header->dim +
         (size_t)header->layers * header->dim +
         (size_t)header->layers * header->hidden_dim * header->dim * 2u +
         (size_t)header->layers * header->dim * header->hidden_dim +
         header->dim +
         (size_t)source_context * head_size +
         ((flags & 1u) == 0u ?
          (size_t)header->vocab_size * header->dim : 0u)) * sizeof(float);
    if (bytes != expected) {
        return LLM_ERR_MODEL_SIZE;
    }
    if (header->checksum != llm_checksum(
            (const uint8_t *)buffer + sizeof(*header),
            bytes - sizeof(*header))) {
        return LLM_ERR_MODEL_FORMAT;
    }

    model->dim = header->dim;
    model->hidden_dim = header->hidden_dim;
    model->layers = header->layers;
    model->heads = header->heads;
    model->kv_heads = header->kv_heads;
    model->vocab_size = header->vocab_size;
    model->context_length = header->seq_flags & 0xFFFFu;
    if (model->context_length > LLM_CONTEXT_LIMIT) {
        model->context_length = LLM_CONTEXT_LIMIT;
    }
    model->shared_classifier = flags & 1u;
    model->kv_dim = kv_dim;
    weights = (const float *)((const uint8_t *)buffer + sizeof(*header));
    model->embedding = weights;
    weights += (size_t)model->vocab_size * model->dim;
    model->rms_att = weights;
    weights += (size_t)model->layers * model->dim;
    model->wq = weights;
    weights += (size_t)model->layers * model->dim * model->dim;
    model->wk = weights;
    weights += (size_t)model->layers * model->dim * model->kv_dim;
    model->wv = weights;
    weights += (size_t)model->layers * model->dim * model->kv_dim;
    model->wo = weights;
    weights += (size_t)model->layers * model->dim * model->dim;
    model->rms_ffn = weights;
    weights += (size_t)model->layers * model->dim;
    model->w1 = weights;
    weights += (size_t)model->layers * model->hidden_dim * model->dim;
    model->w2 = weights;
    weights += (size_t)model->layers * model->dim * model->hidden_dim;
    model->w3 = weights;
    weights += (size_t)model->layers * model->hidden_dim * model->dim;
    model->rms_final = weights;
    weights += model->dim;
    weights += (size_t)source_context * head_size;
    model->classifier = model->shared_classifier ?
        model->embedding : weights;
    return LLM_OK;
}

static llm_status_t llm_attach_tokenizer(llm_tokenizer_t *tokenizer,
                                         const void *buffer, size_t bytes,
                                         uint32_t expected_vocab,
                                         uint8_t *arena, size_t capacity,
                                         size_t *cursor)
{
    const llm_tokenizer_header_t *header =
        (const llm_tokenizer_header_t *)buffer;
    const uint8_t *record;
    size_t required;
    uint32_t index;

    if (buffer == 0 || bytes < sizeof(*header)) {
        return LLM_ERR_TOKENIZER_SIZE;
    }
    if (header->magic != LLM_TOKENIZER_MAGIC ||
        header->header_bytes != sizeof(*header)) {
        return LLM_ERR_TOKENIZER_FORMAT;
    }
    if (header->version != LLM_FORMAT_VERSION) {
        return LLM_ERR_TOKENIZER_VERSION;
    }
    if (header->vocab_size != expected_vocab ||
        header->max_token_length == 0 ||
        header->records_bytes != bytes - sizeof(*header)) {
        return LLM_ERR_TOKENIZER_FORMAT;
    }
    if (header->checksum != llm_checksum(
            (const uint8_t *)buffer + sizeof(*header),
            header->records_bytes)) {
        return LLM_ERR_TOKENIZER_FORMAT;
    }
    required = llm_align(*cursor) +
        (size_t)header->vocab_size * sizeof(uint8_t *) +
        (size_t)header->vocab_size * sizeof(uint32_t);
    if (required > capacity) {
        return LLM_ERR_ARENA_EXHAUSTED;
    }
    *cursor = llm_align(*cursor);
    tokenizer->token_data = (const uint8_t **)(void *)(arena + *cursor);
    *cursor += (size_t)header->vocab_size * sizeof(uint8_t *);
    tokenizer->token_lengths = (uint32_t *)(void *)(arena + *cursor);
    *cursor += (size_t)header->vocab_size * sizeof(uint32_t);
    record = (const uint8_t *)buffer + sizeof(*header);
    for (index = 0; index < header->vocab_size; ++index) {
        float score;
        uint32_t length;

        if ((size_t)(record - (const uint8_t *)buffer) + 8u > bytes) {
            return LLM_ERR_TOKENIZER_SIZE;
        }
        __builtin_memcpy(&score, record, sizeof(score));
        __builtin_memcpy(&length, record + 4u, sizeof(length));
        record += 8u;
        if (length > header->max_token_length ||
            (size_t)(record - (const uint8_t *)buffer) + length > bytes) {
            return LLM_ERR_TOKENIZER_FORMAT;
        }
        tokenizer->token_data[index] = record;
        tokenizer->token_lengths[index] = length;
        record += length;
        (void)score;
    }
    if (record != (const uint8_t *)buffer + bytes) {
        return LLM_ERR_TOKENIZER_SIZE;
    }
    tokenizer->vocab_size = header->vocab_size;
    tokenizer->max_token_length = header->max_token_length;
    return LLM_OK;
}

static void *llm_reserve(llm_runtime_t *runtime, size_t bytes)
{
    size_t start = llm_align(runtime->arena_used);

    if (start > runtime->arena_capacity ||
        bytes > runtime->arena_capacity - start) {
        return 0;
    }
    runtime->arena_used = start + bytes;
    if (runtime->arena_used > runtime->arena_peak) {
        runtime->arena_peak = runtime->arena_used;
    }
    return runtime->arena + start;
}

static uint32_t llm_tokenize(const llm_runtime_t *runtime, const char *prompt,
                             size_t bytes, uint32_t *tokens)
{
    uint32_t count = 0;
    size_t offset = 0;

    while (offset < bytes && count < LLM_CONTEXT_LIMIT) {
        uint32_t best = 0;
        uint32_t best_length = 0;
        uint32_t token;

        for (token = 0; token < runtime->tokenizer.vocab_size; ++token) {
            uint32_t length = runtime->tokenizer.token_lengths[token];
            uint32_t index;
            int matches = length != 0 && offset + length <= bytes;

            for (index = 0; matches && index < length; ++index) {
                if ((uint8_t)prompt[offset + index] !=
                    runtime->tokenizer.token_data[token][index]) {
                    matches = 0;
                }
            }
            if (matches && length > best_length) {
                best = token;
                best_length = length;
            }
        }
        if (best_length == 0u) {
            best = 0u;
            best_length = 1u;
        }
        tokens[count++] = best;
        offset += best_length;
    }
    return count;
}

static void llm_forward(llm_runtime_t *runtime, uint32_t token,
                        uint32_t position)
{
    const llm_model_t *model = &runtime->model;
    uint32_t layer;
    uint32_t head_size = model->dim / model->heads;
    uint32_t kv_head_size = model->kv_dim / model->kv_heads;

    llm_copy(runtime->x, model->embedding + (size_t)token * model->dim,
             model->dim);
    for (layer = 0; layer < model->layers; ++layer) {
        const float *rms_att = model->rms_att +
            (size_t)layer * model->dim;
        const float *wq = model->wq +
            (size_t)layer * model->dim * model->dim;
        const float *wk = model->wk +
            (size_t)layer * model->dim * model->kv_dim;
        const float *wv = model->wv +
            (size_t)layer * model->dim * model->kv_dim;
        const float *wo = model->wo +
            (size_t)layer * model->dim * model->dim;
        const float *rms_ffn = model->rms_ffn +
            (size_t)layer * model->dim;
        const float *w1 = model->w1 +
            (size_t)layer * model->hidden_dim * model->dim;
        const float *w2 = model->w2 +
            (size_t)layer * model->dim * model->hidden_dim;
        const float *w3 = model->w3 +
            (size_t)layer * model->hidden_dim * model->dim;
        float *keys = runtime->key_cache +
            ((size_t)layer * model->context_length + position) *
            model->kv_dim;
        float *values = runtime->value_cache +
            ((size_t)layer * model->context_length + position) *
            model->kv_dim;
        uint32_t head;

        llm_rmsnorm(runtime->norm, runtime->x, rms_att, model->dim);
        llm_matvec(runtime->q, wq, runtime->norm, model->dim, model->dim);
        llm_matvec(keys, wk, runtime->norm, model->kv_dim, model->dim);
        llm_matvec(values, wv, runtime->norm, model->kv_dim, model->dim);
        for (head = 0; head < model->heads; ++head) {
            llm_rope(runtime->q + head * head_size, head_size, position,
                     head_size);
        }
        for (head = 0; head < model->kv_heads; ++head) {
            llm_rope(keys + head * kv_head_size, kv_head_size, position,
                     kv_head_size);
        }
        llm_zero(runtime->next, model->dim);
        for (head = 0; head < model->heads; ++head) {
            uint32_t kv_head = head * model->kv_heads / model->heads;
            uint32_t time;
            float maximum = -3.402823466e+38f;

            for (time = 0; time <= position; ++time) {
                const float *past_key = runtime->key_cache +
                    ((size_t)layer * model->context_length + time) *
                    model->kv_dim + kv_head * kv_head_size;
                float score = 0.0f;
                uint32_t index;

                for (index = 0; index < head_size; ++index) {
                    score += runtime->q[head * head_size + index] *
                        past_key[index];
                }
                score *= llm_rsqrt((float)head_size);
                runtime->att[time] = score;
                if (score > maximum) {
                    maximum = score;
                }
            }
            {
                float total = 0.0f;

                for (time = 0; time <= position; ++time) {
                    runtime->att[time] = llm_exp(runtime->att[time] - maximum);
                    total += runtime->att[time];
                }
                total = total > 0.0f ? 1.0f / total : 0.0f;
                for (time = 0; time <= position; ++time) {
                    const float *past_value = runtime->value_cache +
                        ((size_t)layer * model->context_length + time) *
                        model->kv_dim + kv_head * kv_head_size;
                    uint32_t index;

                    for (index = 0; index < head_size; ++index) {
                        runtime->next[head * head_size + index] +=
                            runtime->att[time] * total * past_value[index];
                    }
                }
            }
        }
        llm_matvec(runtime->norm, wo, runtime->next, model->dim, model->dim);
        for (uint32_t index = 0; index < model->dim; ++index) {
            runtime->x[index] += runtime->norm[index];
        }
        llm_rmsnorm(runtime->norm, runtime->x, rms_ffn, model->dim);
        llm_matvec(runtime->ffn_hidden, w1, runtime->norm,
                   model->hidden_dim, model->dim);
        llm_matvec(runtime->ffn_gate, w3, runtime->norm,
                   model->hidden_dim, model->dim);
        for (uint32_t index = 0; index < model->hidden_dim; ++index) {
            float gate = runtime->ffn_hidden[index];

            runtime->ffn_hidden[index] =
                gate / (1.0f + llm_exp(-gate)) * runtime->ffn_gate[index];
        }
        llm_matvec(runtime->norm, w2, runtime->ffn_hidden,
                   model->dim, model->hidden_dim);
        for (uint32_t index = 0; index < model->dim; ++index) {
            runtime->x[index] += runtime->norm[index];
        }
    }
    llm_rmsnorm(runtime->norm, runtime->x, model->rms_final, model->dim);
    llm_matvec(runtime->logits, model->classifier, runtime->norm,
               model->vocab_size, model->dim);
}

static uint32_t llm_argmax(const float *values, uint32_t count)
{
    uint32_t best = 0;
    float score = -3.402823466e+38f;
    uint32_t index;

    for (index = 0; index < count; ++index) {
        if (values[index] > score) {
            score = values[index];
            best = index;
        }
    }
    return best;
}

llm_status_t llm_runtime_init(llm_runtime_t *runtime, void *arena,
                              size_t arena_bytes, const void *model,
                              size_t model_bytes, const void *tokenizer,
                              size_t tokenizer_bytes)
{
    llm_status_t status;
    size_t cursor;
    uint64_t validation_start;

    if (runtime == 0) {
        return LLM_ERR_ARGUMENT;
    }
    if (arena == 0) {
        return LLM_ERR_ARENA_NULL;
    }
    if (arena_bytes < LLM_RUNTIME_ARENA_BYTES) {
        return LLM_ERR_ARENA_SIZE;
    }
    runtime->arena = (uint8_t *)arena;
    runtime->arena_capacity = arena_bytes;
    runtime->arena_used = 0;
    runtime->arena_peak = 0;
    validation_start = llm_read_tsc();
    status = llm_attach_model(&runtime->model, model, model_bytes);
    if (status != LLM_OK) {
        return status;
    }
    cursor = 0;
    status = llm_attach_tokenizer(&runtime->tokenizer, tokenizer,
                                  tokenizer_bytes, runtime->model.vocab_size,
                                  runtime->arena, arena_bytes, &cursor);
    if (status != LLM_OK) {
        return status;
    }
    runtime->arena_used = cursor;
    runtime->arena_peak = cursor;
    runtime->token_ids = (uint32_t *)llm_reserve(
        runtime, (LLM_CONTEXT_LIMIT + LLM_OUTPUT_LIMIT) * sizeof(uint32_t));
    runtime->x = (float *)llm_reserve(runtime,
                                      runtime->model.dim * sizeof(float));
    runtime->next = (float *)llm_reserve(runtime,
                                         runtime->model.dim * sizeof(float));
    runtime->norm = (float *)llm_reserve(runtime,
                                         runtime->model.dim * sizeof(float));
    runtime->q = (float *)llm_reserve(runtime,
                                      runtime->model.dim * sizeof(float));
    runtime->k = (float *)llm_reserve(runtime,
                                      runtime->model.kv_dim * sizeof(float));
    runtime->v = (float *)llm_reserve(runtime,
                                      runtime->model.kv_dim * sizeof(float));
    runtime->att = (float *)llm_reserve(
        runtime, runtime->model.context_length * sizeof(float));
    runtime->ffn_hidden = (float *)llm_reserve(
        runtime, runtime->model.hidden_dim * sizeof(float));
    runtime->ffn_gate = (float *)llm_reserve(
        runtime, runtime->model.hidden_dim * sizeof(float));
    runtime->logits = (float *)llm_reserve(
        runtime, runtime->model.vocab_size * sizeof(float));
    runtime->key_cache = (float *)llm_reserve(
        runtime, (size_t)runtime->model.layers *
            runtime->model.context_length * runtime->model.kv_dim *
            sizeof(float));
    runtime->value_cache = (float *)llm_reserve(
        runtime, (size_t)runtime->model.layers *
            runtime->model.context_length * runtime->model.kv_dim *
            sizeof(float));
    if (runtime->token_ids == 0 || runtime->x == 0 || runtime->next == 0 ||
        runtime->norm == 0 || runtime->q == 0 || runtime->k == 0 ||
        runtime->v == 0 || runtime->att == 0 || runtime->ffn_hidden == 0 ||
        runtime->ffn_gate == 0 || runtime->logits == 0 ||
        runtime->key_cache == 0 || runtime->value_cache == 0) {
        return LLM_ERR_ARENA_EXHAUSTED;
    }
    runtime->metrics = (llm_metrics_t){0};
    runtime->metrics.arena_capacity = arena_bytes;
    runtime->metrics.arena_used = runtime->arena_used;
    runtime->metrics.arena_peak = runtime->arena_peak;
    runtime->metrics.validation_cycles =
        llm_read_tsc() - validation_start;
    return LLM_OK;
}

llm_status_t llm_generate(llm_runtime_t *runtime, const char *prompt,
                          size_t prompt_bytes, uint32_t max_tokens,
                          llm_token_callback_t callback, void *context)
{
    uint64_t start;
    uint64_t tokenization_start;
    uint32_t prompt_tokens;
    uint32_t position;
    uint32_t generated;

    if (runtime == 0 || prompt == 0 || callback == 0) {
        return LLM_ERR_ARGUMENT;
    }
    if (runtime->arena == 0) {
        return LLM_ERR_NOT_INITIALIZED;
    }
    if (prompt_bytes > LLM_CONTEXT_LIMIT ||
        max_tokens == 0 || max_tokens > LLM_OUTPUT_LIMIT) {
        return prompt_bytes > LLM_CONTEXT_LIMIT ?
            LLM_ERR_PROMPT_TOO_LONG : LLM_ERR_OUTPUT_LIMIT;
    }
    start = llm_read_tsc();
    tokenization_start = start;
    prompt_tokens = llm_tokenize(runtime, prompt, prompt_bytes,
                                 runtime->token_ids);
    runtime->metrics.tokenization_cycles =
        llm_read_tsc() - tokenization_start;
    runtime->metrics.prompt_tokens = prompt_tokens;
    if (prompt_tokens == 0u) {
        runtime->token_ids[0] = 0u;
        prompt_tokens = 1u;
    }
    for (position = 0; position < prompt_tokens; ++position) {
        uint64_t prefill_start = llm_read_tsc();

        llm_forward(runtime, runtime->token_ids[position], position);
        runtime->metrics.prefill_cycles += llm_read_tsc() - prefill_start;
    }
    generated = 0;
    position = prompt_tokens;
    while (generated < max_tokens && position < runtime->model.context_length) {
        uint64_t token_start = llm_read_tsc();
        uint32_t token = llm_argmax(runtime->logits, runtime->model.vocab_size);
        const uint8_t *bytes = runtime->tokenizer.token_data[token];
        size_t length = runtime->tokenizer.token_lengths[token];

        if (callback(bytes, length, generated, context) != 0) {
            return LLM_ERR_CALLBACK;
        }
        ++generated;
        runtime->metrics.last_token_cycles = llm_read_tsc() - token_start;
        if (generated < max_tokens && position < runtime->model.context_length) {
            llm_forward(runtime, token, position);
            ++position;
        }
    }
    runtime->metrics.generated_tokens = generated;
    runtime->metrics.generation_cycles = llm_read_tsc() - start;
    runtime->metrics.arena_used = runtime->arena_used;
    runtime->metrics.arena_peak = runtime->arena_peak;
    return LLM_OK;
}

llm_status_t llm_get_metrics(const llm_runtime_t *runtime,
                             llm_metrics_t *metrics)
{
    if (runtime == 0 || metrics == 0) {
        return LLM_ERR_ARGUMENT;
    }
    if (runtime->arena == 0) {
        return LLM_ERR_NOT_INITIALIZED;
    }
    *metrics = runtime->metrics;
    return LLM_OK;
}

const char *llm_status_string(llm_status_t status)
{
    switch (status) {
    case LLM_OK:
        return "ok";
    case LLM_ERR_ARGUMENT:
        return "invalid argument";
    case LLM_ERR_MODEL_SIZE:
        return "invalid model size";
    case LLM_ERR_MODEL_FORMAT:
        return "invalid model format";
    case LLM_ERR_MODEL_VERSION:
        return "unsupported model version";
    case LLM_ERR_TOKENIZER_SIZE:
        return "invalid tokenizer size";
    case LLM_ERR_TOKENIZER_FORMAT:
        return "invalid tokenizer format";
    case LLM_ERR_TOKENIZER_VERSION:
        return "unsupported tokenizer version";
    case LLM_ERR_ARENA_NULL:
        return "arena is null";
    case LLM_ERR_ARENA_SIZE:
        return "arena is too small";
    case LLM_ERR_ARENA_EXHAUSTED:
        return "arena exhausted";
    case LLM_ERR_PROMPT_TOO_LONG:
        return "prompt is too long";
    case LLM_ERR_OUTPUT_LIMIT:
        return "output limit is invalid";
    case LLM_ERR_CALLBACK:
        return "token callback failed";
    case LLM_ERR_NOT_INITIALIZED:
        return "runtime not initialized";
    default:
        return "unknown runtime error";
    }
}
