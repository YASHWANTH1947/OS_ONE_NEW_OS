#ifndef UI_H
#define UI_H

#include <stddef.h>
#include <stdint.h>

#include "llm_runtime.h"

typedef struct {
    uint32_t *pixels;
    uint16_t *text_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t pitch_pixels;
    uint32_t columns;
    uint32_t rows;
    uint32_t column;
    uint32_t row;
    char input[LLM_CONTEXT_LIMIT + 1u];
    uint32_t input_length;
    uint32_t running;
    int text_mode;
    uint64_t framebuffer_address;
    uint32_t framebuffer_pitch;
    uint32_t framebuffer_bpp;
    uint64_t render_cycles;
    int available;
} ui_console_t;

int ui_init(uint32_t multiboot_info, ui_console_t *console);
void ui_write(ui_console_t *console, const uint8_t *bytes, size_t length);
void ui_clear(ui_console_t *console);
void ui_keyboard_poll(ui_console_t *console, llm_runtime_t *runtime);
void keyboard_init(void);
void keyboard_irq_handler(void);

#endif
