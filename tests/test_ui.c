#include "ui.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    uint32_t pixels[64u * 32u];
    ui_console_t console = {0};
    uint32_t index;
    int changed = 0;

    console.pixels = pixels;
    console.width = 64u;
    console.height = 32u;
    console.pitch_pixels = 64u;
    console.columns = 8u;
    console.rows = 2u;
    console.available = 1;
    ui_clear(&console);
    for (index = 0; index < 64u * 32u; ++index) {
        assert(pixels[index] == 0x00101828u);
    }
    ui_write(&console, (const uint8_t *)"A\nB", 3u);
    for (index = 0; index < 64u * 32u; ++index) {
        if (pixels[index] != 0x00101828u) {
            changed = 1;
            break;
        }
    }
    assert(changed);
    assert(console.column == 1u);
    assert(console.row == 1u);
    puts("UI tests passed.");
    return 0;
}
