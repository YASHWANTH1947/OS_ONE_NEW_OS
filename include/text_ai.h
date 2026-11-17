#ifndef TEXT_AI_H
#define TEXT_AI_H

#include <stddef.h>
#include <stdint.h>

#define TEXT_AI_MAGIC 0x31494154u /* "TAI1" */
#define TEXT_AI_VERSION 1u
#define TEXT_AI_VOCAB 42u
#define TEXT_AI_HIDDEN 128u
#define TEXT_AI_PROMPT_LIMIT 96u
#define TEXT_AI_OUTPUT_LIMIT 160u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t reserved;
    float input[TEXT_AI_VOCAB * TEXT_AI_HIDDEN];
    float recurrent[TEXT_AI_HIDDEN * TEXT_AI_HIDDEN];
    float output[TEXT_AI_HIDDEN * TEXT_AI_VOCAB];
    float hidden_bias[TEXT_AI_HIDDEN];
    float output_bias[TEXT_AI_VOCAB];
} text_ai_model_t;

int text_ai_model_valid(const text_ai_model_t *model, size_t bytes);

#endif
