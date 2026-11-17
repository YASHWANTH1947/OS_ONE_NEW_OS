#include "ui.h"

#include "kernel.h"

#define MULTIBOOT_TAG_END 0u
#define MULTIBOOT_TAG_FRAMEBUFFER 8u
#define KEYBOARD_DATA 0x60u
#define KEYBOARD_STATUS 0x64u
#define PIC_MASTER_COMMAND 0x20u
#define PIC_MASTER_DATA 0x21u
#define PIC_SLAVE_COMMAND 0xA0u
#define PIC_SLAVE_DATA 0xA1u
#define PIC_EOI 0x20u

typedef struct {
    uint32_t type;
    uint32_t size;
} multiboot_tag_t;

typedef struct {
    uint32_t type;
    uint32_t size;
    uint64_t address;
    uint32_t pitch;
    uint32_t width;
    uint32_t height;
    uint8_t bpp;
    uint8_t type_info;
    uint16_t reserved;
    uint8_t color_data[6];
} __attribute__((packed)) framebuffer_tag_t;

static uint8_t key_queue[128];
static uint32_t key_read;
static uint32_t key_write;
static int shift_down;

static const uint8_t ui_banner[] =
    "============================================================\n"
    " AI OS - TinyStories local inference\n"
    " Commands: help status ask TEXT clear shutdown\n"
    " Type text and press ENTER to ask the local AI.\n"
    " ESC exits QEMU.\n"
    "============================================================\n"
    "AI> ";

static uint64_t ui_read_tsc(void)
{
    uint32_t low;
    uint32_t high;

    __asm__ volatile ("lfence; rdtsc; lfence"
                      : "=a"(low), "=d"(high) : : "memory");
    return ((uint64_t)high << 32) | low;
}

static const uint8_t font[36][5] = {
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36},
    {0x3E, 0x41, 0x41, 0x41, 0x22}, {0x7F, 0x41, 0x41, 0x22, 0x1C},
    {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x09, 0x01},
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, {0x7F, 0x08, 0x08, 0x08, 0x7F},
    {0x00, 0x41, 0x7F, 0x41, 0x00}, {0x20, 0x40, 0x41, 0x3F, 0x01},
    {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x04, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F},
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, {0x7F, 0x09, 0x09, 0x09, 0x06},
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01},
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, {0x1F, 0x20, 0x40, 0x20, 0x1F},
    {0x7F, 0x20, 0x18, 0x20, 0x7F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x07, 0x08, 0x70, 0x08, 0x07}, {0x61, 0x51, 0x49, 0x45, 0x43},
    {0x3E, 0x45, 0x49, 0x51, 0x3E}, {0x00, 0x21, 0x7F, 0x01, 0x00},
    {0x23, 0x45, 0x49, 0x51, 0x31}, {0x42, 0x41, 0x51, 0x69, 0x46},
    {0x0C, 0x14, 0x24, 0x7F, 0x04}, {0x72, 0x51, 0x51, 0x51, 0x4E},
    {0x1E, 0x29, 0x49, 0x49, 0x06}, {0x40, 0x47, 0x48, 0x50, 0x60},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x30, 0x49, 0x49, 0x4A, 0x3C}
};

static void ui_pixel(ui_console_t *console, uint32_t x, uint32_t y,
                     uint32_t color)
{
    if (!console->available || console->text_mode ||
        x >= console->width || y >= console->height) {
        return;
    }
    console->pixels[y * console->pitch_pixels + x] = color;
}

static void ui_text_put(ui_console_t *console, char character)
{
    if (console->text_buffer == 0 ||
        console->column >= console->columns ||
        console->row >= console->rows) {
        return;
    }
    console->text_buffer[console->row * console->columns + console->column] =
        (uint16_t)(0x1Fu << 8) | (uint8_t)character;
    ++console->column;
    if (console->column >= console->columns) {
        console->column = 0;
        ++console->row;
    }
}

static int glyph_index(char character)
{
    if (character >= 'a' && character <= 'z') {
        character = (char)(character - 'a' + 'A');
    }
    if (character >= 'A' && character <= 'Z') {
        return character - 'A';
    }
    if (character >= '0' && character <= '9') {
        return 26 + character - '0';
    }
    return -1;
}

static void ui_glyph(ui_console_t *console, char character)
{
    int index = glyph_index(character);
    uint32_t row;
    uint32_t column;

    for (row = 0; row < 16u; ++row) {
        for (column = 0; column < 8u; ++column) {
            int set = 0;

            if (index >= 0 && column < 5u) {
                set = (font[index][column] & (1u << (row / 2u))) != 0;
            } else if (character != ' ') {
                set = (column == 0u || column == 7u ||
                       row == 0u || row == 15u);
            }
            ui_pixel(console, console->column * 8u + column,
                     console->row * 16u + row, set ? 0x00FFFFFFu : 0x00101828u);
        }
    }
    ++console->column;
    if (console->column >= console->columns) {
        console->column = 0;
        ++console->row;
    }
    if (console->row >= console->rows) {
        console->row = console->rows - 1u;
        ui_clear(console);
    }
}

int ui_init(uint32_t multiboot_info, ui_console_t *console)
{
    const uint8_t *base = (const uint8_t *)(uintptr_t)multiboot_info;
    const uint32_t total_size = *(const uint32_t *)base;
    uint32_t offset = 8u;

    if (console == 0) {
        return 0;
    }
    console->available = 0;
    console->text_mode = 0;
    console->text_buffer = 0;
    console->framebuffer_address = 0;
    console->framebuffer_pitch = 0;
    console->framebuffer_bpp = 0;
    while (offset + sizeof(multiboot_tag_t) <= total_size) {
        const multiboot_tag_t *tag =
            (const multiboot_tag_t *)(base + offset);

        if (tag->size < sizeof(*tag) || offset + tag->size > total_size) {
            return 0;
        }
        if (tag->type == MULTIBOOT_TAG_END) {
            break;
        }
        if (tag->type == MULTIBOOT_TAG_FRAMEBUFFER &&
            tag->size >= 32u) {
            const framebuffer_tag_t *framebuffer =
                (const framebuffer_tag_t *)tag;

            console->framebuffer_address = framebuffer->address;
            console->framebuffer_pitch = framebuffer->pitch;
            console->framebuffer_bpp = framebuffer->bpp;
            console->width = framebuffer->width;
            console->height = framebuffer->height;
            if (framebuffer->address == 0xB8000u &&
                framebuffer->width == 80u &&
                framebuffer->height == 25u &&
                framebuffer->pitch >= 160u &&
                framebuffer->bpp == 16u) {
                console->text_buffer =
                    (uint16_t *)(uintptr_t)framebuffer->address;
                console->columns = 80u;
                console->rows = 25u;
                console->column = 0;
                console->row = 0;
                console->input_length = 0;
                console->render_cycles = 0;
                console->running = 1;
                console->text_mode = 1;
                console->available = 1;
                ui_clear(console);
                ui_write(console, ui_banner, sizeof(ui_banner) - 1u);
                return 1;
            }
            if (framebuffer->address < (1ULL << 30) &&
                framebuffer->bpp == 32u &&
                framebuffer->width >= 640u &&
                framebuffer->height >= 400u &&
                framebuffer->pitch >= framebuffer->width * 4u) {
                console->pixels =
                    (uint32_t *)(uintptr_t)framebuffer->address;
                console->width = framebuffer->width;
                console->height = framebuffer->height;
                console->pitch_pixels = framebuffer->pitch / 4u;
                console->columns = console->width / 8u;
                console->rows = console->height / 16u;
                console->column = 0;
                console->row = 0;
                console->input_length = 0;
                console->render_cycles = 0;
                console->running = 1;
                console->available = 1;
                ui_clear(console);
                ui_write(console, ui_banner, sizeof(ui_banner) - 1u);
                return 1;
            }
        }
        offset = (offset + tag->size + 7u) & ~7u;
    }
    return 0;
}

void ui_clear(ui_console_t *console)
{
    uint32_t y;
    uint32_t x;

    if (console == 0 || !console->available) {
        return;
    }
    if (console->text_mode) {
        uint32_t index;

        for (index = 0; index < console->columns * console->rows; ++index) {
            console->text_buffer[index] = (uint16_t)(0x1Fu << 8) | ' ';
        }
        console->column = 0;
        console->row = 0;
        return;
    }
    for (y = 0; y < console->height; ++y) {
        for (x = 0; x < console->width; ++x) {
            ui_pixel(console, x, y, 0x00101828u);
        }
    }
    console->column = 0;
    console->row = 0;
}

void ui_write(ui_console_t *console, const uint8_t *bytes, size_t length)
{
    uint64_t start_cycles;
    size_t index;

    if (console == 0 || !console->available || bytes == 0) {
        return;
    }
    start_cycles = ui_read_tsc();
    for (index = 0; index < length; ++index) {
        if (bytes[index] == '\n') {
            console->column = 0;
            ++console->row;
            if (console->row >= console->rows) {
                console->row = 0;
                ui_clear(console);
            }
        } else if (bytes[index] == '\r') {
            console->column = 0;
        } else if (bytes[index] == '\b') {
            if (console->column > 0) {
                --console->column;
                if (console->text_mode) {
                    console->text_buffer[
                        console->row * console->columns + console->column] =
                        (uint16_t)(0x1Fu << 8) | ' ';
                } else {
                    ui_glyph(console, ' ');
                    --console->column;
                }
            }
        } else if (console->text_mode) {
            ui_text_put(console, (char)bytes[index]);
        } else {
            ui_glyph(console, (char)bytes[index]);
        }
        if (console->row >= console->rows) {
            console->row = 0;
            ui_clear(console);
        }
    }
    console->render_cycles += ui_read_tsc() - start_cycles;
}

static void keyboard_push(uint8_t value)
{
    uint32_t next = (key_write + 1u) % sizeof(key_queue);

    if (next != key_read) {
        key_queue[key_write] = value;
        key_write = next;
    }
}

static int ui_text_equals(const char *left, const char *right)
{
    uint32_t index = 0;

    while (left[index] != '\0' && right[index] != '\0') {
        if (left[index] != right[index]) {
            return 0;
        }
        ++index;
    }
    return left[index] == '\0' && right[index] == '\0';
}

static int ui_text_starts_with(const char *text, const char *prefix)
{
    uint32_t index = 0;

    while (prefix[index] != '\0') {
        if (text[index] != prefix[index]) {
            return 0;
        }
        ++index;
    }
    return 1;
}

static const char *ui_skip_spaces(const char *text)
{
    while (*text == ' ') {
        ++text;
    }
    return text;
}

static void ui_show_help(ui_console_t *console)
{
    static const uint8_t help[] =
        "\nCOMMANDS\n"
        " help       show this list\n"
        " status     show local AI status\n"
        " ask TEXT   generate a response\n"
        " clear      clear the screen\n"
        " shutdown   halt the AI OS\n"
        " ESC        halt the AI OS\n";

    ui_write(console, help, sizeof(help) - 1u);
}

static void ui_show_status(ui_console_t *console,
                           const llm_runtime_t *runtime)
{
    static const uint8_t status[] =
        "\nSTATUS\n"
        " mode       VGA keyboard console\n"
        " model      local TinyStories\n"
        " inference  ready\n";

    ui_write(console, status, sizeof(status) - 1u);
    if (runtime != 0) {
        static const uint8_t metrics[] = " generations completed; cycles: ";
        char digits[24];
        uint64_t value = runtime->metrics.generation_cycles;
        uint32_t length = 0;

        ui_write(console, metrics, sizeof(metrics) - 1u);
        do {
            digits[length++] = (char)('0' + (value % 10u));
            value /= 10u;
        } while (value != 0 && length < sizeof(digits));
        while (length != 0) {
            --length;
            ui_write(console, (const uint8_t *)&digits[length], 1u);
        }
        ui_write(console, (const uint8_t *)"\n", 1u);
    }
}

void keyboard_irq_handler(void)
{
    static const char normal[128] = {
        [0x02] = '1', [0x03] = '2', [0x04] = '3', [0x05] = '4',
        [0x06] = '5', [0x07] = '6', [0x08] = '7', [0x09] = '8',
        [0x0A] = '9', [0x0B] = '0', [0x10] = 'q', [0x11] = 'w',
        [0x12] = 'e', [0x13] = 'r', [0x14] = 't', [0x15] = 'y',
        [0x16] = 'u', [0x17] = 'i', [0x18] = 'o', [0x19] = 'p',
        [0x1E] = 'a', [0x1F] = 's', [0x20] = 'd', [0x21] = 'f',
        [0x22] = 'g', [0x23] = 'h', [0x24] = 'j', [0x25] = 'k',
        [0x26] = 'l', [0x2C] = 'z', [0x2D] = 'x', [0x2E] = 'c',
        [0x2F] = 'v', [0x30] = 'b', [0x31] = 'n', [0x32] = 'm',
        [0x39] = ' '
    };
    static int extended_scan;
    uint8_t scan = port_in(KEYBOARD_DATA);

    if (scan == 0xE0u || scan == 0xE1u) {
        extended_scan = 1;
        port_out(PIC_MASTER_COMMAND, PIC_EOI);
        return;
    }
    if (extended_scan != 0) {
        extended_scan = 0;
        port_out(PIC_MASTER_COMMAND, PIC_EOI);
        return;
    }
    if ((scan & 0x80u) != 0) {
        if ((scan & 0x7Fu) == 0x2Au || (scan & 0x7Fu) == 0x36u) {
            shift_down = 0;
        }
    } else if (scan == 0x2Au || scan == 0x36u) {
        shift_down = 1;
    } else if (scan == 0x1Cu) {
        keyboard_push('\n');
    } else if (scan == 0x01u) {
        keyboard_push(0x1Bu);
    } else if (scan == 0x0Eu) {
        keyboard_push('\b');
    } else if (scan < sizeof(normal) && normal[scan] != '\0') {
        char value = normal[scan];

        if (shift_down && value >= 'a' && value <= 'z') {
            value = (char)(value - 'a' + 'A');
        }
        keyboard_push((uint8_t)value);
    }
    port_out(PIC_MASTER_COMMAND, PIC_EOI);
}

static int ui_llm_token(const uint8_t *bytes, size_t length,
                        uint32_t token_index, void *context)
{
    (void)token_index;
    ui_write((ui_console_t *)context, bytes, length);
    return 0;
}

void keyboard_init(void)
{
    uint8_t mask;

    port_out(PIC_MASTER_COMMAND, 0x11u);
    port_out(PIC_SLAVE_COMMAND, 0x11u);
    port_out(PIC_MASTER_DATA, 0x20u);
    port_out(PIC_SLAVE_DATA, 0x28u);
    port_out(PIC_MASTER_DATA, 0x04u);
    port_out(PIC_SLAVE_DATA, 0x02u);
    port_out(PIC_MASTER_DATA, 0x01u);
    port_out(PIC_SLAVE_DATA, 0x01u);
    mask = port_in(PIC_MASTER_DATA);
    port_out(PIC_MASTER_DATA, (uint8_t)((mask | 0x01u) & ~0x02u));
    port_out(PIC_SLAVE_DATA, 0xFFu);
    while ((port_in(KEYBOARD_STATUS) & 1u) != 0u) {
        (void)port_in(KEYBOARD_DATA);
    }
}

void ui_keyboard_poll(ui_console_t *console, llm_runtime_t *runtime)
{
    while (key_read != key_write && console->running) {
        uint8_t value = key_queue[key_read];

        key_read = (key_read + 1u) % sizeof(key_queue);
        if (value == '\n') {
            static const uint8_t prompt_prefix[] = "\nAI> ";
            char *line = console->input;
            const char *prompt = line;
            uint32_t index;

            line[console->input_length] = '\0';
            ui_write(console, (const uint8_t *)"\n", 1u);
            if (ui_text_equals(line, "help")) {
                ui_show_help(console);
            } else if (ui_text_equals(line, "status")) {
                ui_show_status(console, runtime);
            } else if (ui_text_equals(line, "clear")) {
                ui_clear(console);
                ui_write(console, ui_banner, sizeof(ui_banner) - 1u);
            } else if (ui_text_equals(line, "shutdown") ||
                       ui_text_equals(line, "exit")) {
                ui_write(console, (const uint8_t *)"[shutdown]\n", 11u);
                console->running = 0;
            } else {
                if (ui_text_starts_with(line, "ask")) {
                    if (line[3] == '\0' || line[3] == ' ') {
                        prompt = ui_skip_spaces(line + 3);
                    }
                }
                if (runtime != 0 && prompt[0] != '\0') {
                    llm_generate(runtime, prompt,
                                 (uint32_t)(console->input_length -
                                            (prompt - line)),
                                 64u, ui_llm_token, console);
                }
            }
            console->input_length = 0;
            for (index = 0; index < sizeof(console->input); ++index) {
                console->input[index] = '\0';
            }
            if (console->running) {
                ui_write(console, prompt_prefix, sizeof(prompt_prefix) - 1u);
            }
        } else if (value == '\b') {
            if (console->input_length != 0) {
                --console->input_length;
                ui_write(console, (const uint8_t *)"\b", 1u);
            }
        } else if (value == 0x1Bu) {
            console->running = 0;
            ui_write(console, (const uint8_t *)"\n[shutdown]\n", 12u);
        } else if (console->input_length < LLM_CONTEXT_LIMIT) {
            console->input[console->input_length++] = (char)value;
            ui_write(console, &value, 1u);
        }
    }
}
