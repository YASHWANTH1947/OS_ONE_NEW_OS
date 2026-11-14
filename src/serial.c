#include "kernel.h"

#include <stdarg.h>

/*
 * COM1 driver plus a small formatting layer. Everything the kernel prints
 * goes through here so the boot transcript stays one consistent, readable
 * transcript with no dependency on a C library.
 */

#define COM1 0x3F8u
#define CONSOLE_WIDTH 74u

static void serial_putc(char value)
{
    uint32_t timeout = 100000u;

    while ((port_in(COM1 + 5u) & 0x20u) == 0 && timeout != 0) {
        --timeout;
    }
    if (timeout != 0) {
        port_out(COM1, (uint8_t)value);
    }
}

void serial_init(void)
{
    port_out(COM1 + 1u, 0x00u); /* disable interrupts */
    port_out(COM1 + 3u, 0x80u); /* enable divisor latch */
    port_out(COM1 + 0u, 0x01u); /* 115200 baud */
    port_out(COM1 + 1u, 0x00u);
    port_out(COM1 + 3u, 0x03u); /* 8N1 */
    port_out(COM1 + 2u, 0xC7u); /* FIFO enable, clear, 14-byte threshold */
    port_out(COM1 + 4u, 0x0Bu); /* DTR, RTS, OUT2 */
}

void serial_write(const char *text)
{
    while (*text != '\0') {
        if (*text == '\n') {
            serial_putc('\r');
        }
        serial_putc(*text++);
    }
}

int serial_available(void)
{
    return (port_in(COM1 + 5u) & 0x01u) != 0;
}

char serial_read(void)
{
    return (char)port_in(COM1);
}

void serial_repeat(char character, uint32_t count)
{
    char pair[2];

    pair[0] = character;
    pair[1] = '\0';
    while (count != 0) {
        serial_write(pair);
        --count;
    }
}

size_t serial_length(const char *text)
{
    size_t length = 0;

    while (text != 0 && text[length] != '\0') {
        ++length;
    }
    return length;
}

void serial_rule(void)
{
    serial_repeat('-', 78u);
    serial_write("\n");
}

/*
 * Left-aligned label, dot leader, right-hand value. Used for the boot stage
 * table so the transcript reads like a status board rather than log soup.
 */
void serial_aligned(const char *label, const char *value)
{
    size_t label_length = serial_length(label);
    size_t value_length = serial_length(value);

    serial_write("  ");
    serial_write(label);
    if (label_length + value_length + 4u < CONSOLE_WIDTH) {
        serial_write(" ");
        serial_repeat('.', (uint32_t)(CONSOLE_WIDTH - label_length -
                                      value_length - 2u));
        serial_write(" ");
    } else {
        serial_write(" ");
    }
    serial_write(value);
    serial_write("\n");
}

static void print_unsigned(uint64_t value, uint32_t base, int uppercase,
                           uint32_t width, int zero_pad)
{
    char digits[32];
    const char *alphabet = uppercase ? "0123456789ABCDEF" :
                                       "0123456789abcdef";
    uint32_t count = 0;

    do {
        uint64_t quotient = 0;
        uint32_t remainder = 0;
        int bit;

        for (bit = 63; bit >= 0; --bit) {
            remainder = remainder * 2u + (uint32_t)((value >> bit) & 1u);
            if (remainder >= base) {
                remainder -= base;
                quotient |= (uint64_t)1u << bit;
            }
        }
        digits[count++] = alphabet[remainder];
        value = quotient;
    } while (value != 0 && count < sizeof(digits));

    if (width > count) {
        if (zero_pad) {
            while (count < width && count < sizeof(digits)) {
                digits[count++] = '0';
            }
        } else {
            serial_repeat(' ', width - count);
        }
    }
    while (count != 0) {
        serial_putc(digits[--count]);
    }
}

void serial_printf(const char *format, ...)
{
    va_list args;

    va_start(args, format);
    while (*format != '\0') {
        int long_value = 0;
        int long_long_value = 0;
        uint32_t width = 0;
        int zero_pad = 0;

        if (*format != '%') {
            if (*format == '\n') {
                serial_putc('\r');
            }
            serial_putc(*format++);
            continue;
        }
        ++format;
        if (*format == '%') {
            serial_putc('%');
            ++format;
            continue;
        }
        if (*format == '0') {
            zero_pad = 1;
            ++format;
        }
        while (*format >= '0' && *format <= '9') {
            width = width * 10u + (uint32_t)(*format - '0');
            ++format;
        }
        if (*format == 'l') {
            ++format;
            long_value = 1;
            if (*format == 'l') {
                ++format;
                long_long_value = 1;
            }
        }

        switch (*format) {
        case 's': {
            const char *text = va_arg(args, const char *);
            serial_write(text != 0 ? text : "(null)");
            break;
        }
        case 'c':
            serial_putc((char)va_arg(args, int));
            break;
        case 'u':
            if (long_long_value) {
                print_unsigned(va_arg(args, unsigned long long), 10u, 0,
                               width, zero_pad);
            } else if (long_value) {
                print_unsigned(va_arg(args, unsigned long), 10u, 0,
                               width, zero_pad);
            } else {
                print_unsigned(va_arg(args, unsigned int), 10u, 0,
                               width, zero_pad);
            }
            break;
        case 'd': {
            int64_t value = long_long_value ?
                va_arg(args, long long) : (long_value ?
                va_arg(args, long) : va_arg(args, int));
            uint64_t magnitude;

            if (value < 0) {
                serial_putc('-');
                magnitude = (uint64_t)(-(value + 1)) + 1u;
            } else {
                magnitude = (uint64_t)value;
            }
            print_unsigned(magnitude, 10u, 0, width, zero_pad);
            break;
        }
        case 'x':
        case 'X':
            if (long_long_value) {
                print_unsigned(va_arg(args, unsigned long long), 16u,
                               *format == 'X', width, zero_pad);
            } else if (long_value) {
                print_unsigned(va_arg(args, unsigned long), 16u,
                               *format == 'X', width, zero_pad);
            } else {
                print_unsigned(va_arg(args, unsigned int), 16u,
                               *format == 'X', width, zero_pad);
            }
            break;
        case 'p':
            serial_write("0x");
            print_unsigned((uintptr_t)va_arg(args, void *), 16u, 0, 0u, 0);
            break;
        default:
            serial_putc('%');
            serial_putc(*format);
            break;
        }
        if (*format != '\0') {
            ++format;
        }
    }
    va_end(args);
}

void kernel_panic(const char *message)
{
    serial_write("\n  KERNEL PANIC: ");
    serial_write(message);
    serial_write("\n  System halted.\n");
    for (;;) {
        __asm__ volatile ("cli; hlt");
    }
}
