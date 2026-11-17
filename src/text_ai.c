#include "text_ai.h"

int text_ai_model_valid(const text_ai_model_t *model, size_t bytes)
{
    return model != 0 && bytes == sizeof(*model) &&
           model->magic == TEXT_AI_MAGIC &&
           model->version == TEXT_AI_VERSION &&
           model->vocab == TEXT_AI_VOCAB &&
           model->hidden == TEXT_AI_HIDDEN;
}
