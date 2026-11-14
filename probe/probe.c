/*
 * AVX2/FMA capability and relative-speed probe (AEGIS-X System B evidence).
 * Not shipped code: this answers, on the actual deployment environment, the
 * questions the dispatch engine design depends on:
 *
 *   1. Which of SSE2 / AVX / AVX2 / FMA does CPUID advertise here?
 *   2. Does the CPU actually execute them (after enabling OSXSAVE + XCR0,
 *      which a bare-metal kernel must do itself)?
 *   3. At the MLP's two matvec shapes (128x784 and 10x128), what are the
 *      cycle ratios of the AVX2 and FMA kernels against the tuned scalar loop?
 *   4. Do they compute the same answer (max absolute difference)?
 *
 * Prints a "PROBE BEGIN ... PROBE END" block on COM1, then exits QEMU via
 * isa-debug-exit. Links against boot/boot.S, which calls kernel_main().
 */
#include <stdint.h>
#include <stddef.h>

/* SIMD kernels live in separate translation units compiled with their own
 * ISA flags, which is exactly how the shipped dispatch engine is structured. */
extern void matvec_avx2(const float *w, const float *x, const float *b,
                        float *out, uint32_t rows, uint32_t cols);
extern void matvec_fma(const float *w, const float *x, const float *b,
                       float *out, uint32_t rows, uint32_t cols);

typedef void (*matvec_fn)(const float *, const float *, const float *,
                          float *, uint32_t, uint32_t);

/* ---------------------------------------------------------------- serial -- */
#define COM1 0x3F8u

static inline void outb(uint16_t port, uint8_t value)
{
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}

static inline uint8_t inb(uint16_t port)
{
    uint8_t value;

    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}

static void serial_init(void)
{
    outb(COM1 + 1u, 0x00u);
    outb(COM1 + 3u, 0x80u);
    outb(COM1 + 0u, 0x01u);
    outb(COM1 + 1u, 0x00u);
    outb(COM1 + 3u, 0x03u);
    outb(COM1 + 2u, 0xC7u);
    outb(COM1 + 4u, 0x0Bu);
}

static void putc_raw(char value)
{
    uint32_t timeout = 100000u;

    while ((inb(COM1 + 5u) & 0x20u) == 0 && timeout != 0) {
        --timeout;
    }
    if (timeout != 0) {
        outb(COM1, (uint8_t)value);
    }
}

static void puts_line(const char *text)
{
    while (*text != '\0') {
        if (*text == '\n') {
            putc_raw('\r');
        }
        putc_raw(*text++);
    }
}

static void putu(uint64_t value)
{
    char digits[24];
    uint32_t count = 0;

    do {
        digits[count++] = (char)('0' + (value % 10u));
        value /= 10u;
    } while (value != 0);
    while (count != 0) {
        putc_raw(digits[--count]);
    }
}

static void puthex64(uint64_t value)
{
    static const char alphabet[] = "0123456789abcdef";
    char digits[16];
    uint32_t count = 0;
    int bit;

    for (bit = 0; bit < 64; bit += 4, value >>= 4) {
        digits[count++] = alphabet[value & 0xFu];
    }
    putc_raw('0');
    putc_raw('x');
    while (count != 0) {
        putc_raw(digits[--count]);
    }
}

/* Print a small non-negative float as "N.ddd" with the given scale (power of
 * ten by which the value was pre-multiplied). Integer-only after the one
 * float->uint32 conversion, which compiles to a single cvttss2si. */
static void put_scaled(uint32_t scaled, uint32_t decimals)
{
    uint32_t scale = 1;
    uint32_t i;

    for (i = 0; i < decimals; ++i) {
        scale *= 10u;
    }
    putu(scaled / scale);
    putc_raw('.');
    for (i = scale / 10u; i >= 1u; i /= 10u) {
        putc_raw((char)('0' + (scaled / i) % 10u));
    }
}

/* ----------------------------------------------------------------- cpuid -- */
static void cpuid_read(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    __asm__ volatile ("cpuid"
                      : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                      : "a"(leaf), "c"(subleaf));
}

static void print_brand(void)
{
    char brand[49];
    const char *cursor = brand;
    uint32_t regs[4];
    uint32_t line;
    uint32_t i;

    for (line = 0; line < 3u; ++line) {
        cpuid_read(0x80000002u + line, 0u, regs);
        for (i = 0; i < 4u; ++i) {
            ((uint32_t *)brand)[line * 4u + i] = regs[i];
        }
    }
    brand[48] = '\0';
    while (*cursor == ' ') {
        ++cursor;
    }
    puts_line("  brand ........ ");
    puts_line(cursor);
    puts_line("\n");
}

/* -------------------------------------------------- AVX state enablement -- */
/* A bare-metal kernel must itself enable the extended-state machinery before
 * any YMM instruction is legal: CR4.OSXAVE, then XCR0 with x87+SSE+AVX bits. */
static int enable_avx_state(uint64_t *xcr0_readback)
{
    uint32_t regs[4];
    uint32_t confirm[4];
    uint64_t cr4;

    cpuid_read(1u, 0u, regs);
    /* ECX bit 27 (OSXSAVE) is dynamic: it mirrors CR4, so it cannot gate the
     * capability check. The static gates are bit 26 (XSAVE) and bit 28 (AVX);
     * bit 27 only reports 1 after CR4.OSXSAVE has been set. */
    if ((regs[2] & (1u << 26)) == 0 || (regs[2] & (1u << 28)) == 0) {
        *xcr0_readback = 0;
        return 0;
    }
    __asm__ volatile ("movq %%cr4, %0" : "=r"(cr4));
    cr4 |= 0x40000ULL;              /* CR4.OSXSAVE, bit 18 */
    __asm__ volatile ("movq %0, %%cr4" : : "r"(cr4) : "memory");
    __asm__ volatile ("xsetbv" : : "a"(0x7u), "d"(0u), "c"(0u) : "memory");
    cpuid_read(1u, 0u, confirm);
    puts_line("  cr4_after=");
    puthex64(cr4);
    puts_line(" osxsave_now=");
    putu((confirm[2] >> 27) & 1u);
    puts_line("\n");
    {
        uint32_t low;
        uint32_t high;

        __asm__ volatile ("xgetbv" : "=a"(low), "=d"(high) : "c"(0u));
        *xcr0_readback = ((uint64_t)high << 32) | low;
    }
    return (*xcr0_readback & 0x7u) == 0x7u;
}

/* ------------------------------------------------------------------- pit -- */
/* Identical calibration approach to the shipped kernel: PIT channel 2. */
#define PIT_CH2 0x42u
#define PIT_CMD 0x43u
#define SPKR 0x61u
#define PIT_FREQ 1193182u
#define WINDOW_MS 50u

static uint64_t read_tsc(void)
{
    uint32_t lo;
    uint32_t hi;

    __asm__ volatile ("lfence; rdtsc; lfence" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t tsc_calibrate(void)
{
    uint32_t divisor = (PIT_FREQ / 1000u) * WINDOW_MS;
    uint8_t gate = (uint8_t)(inb(SPKR) & 0xFCu);
    uint32_t polls = 0;
    uint64_t start;
    uint64_t elapsed;

    outb(SPKR, gate);
    outb(PIT_CMD, 0xB0u);
    outb(PIT_CH2, (uint8_t)(divisor & 0xFFu));
    outb(PIT_CH2, (uint8_t)((divisor >> 8) & 0xFFu));
    outb(SPKR, (uint8_t)(gate | 0x01u));
    start = read_tsc();
    while ((inb(SPKR) & 0x20u) == 0) {
        if (++polls > 5000000u) {
            outb(SPKR, gate);
            return 0;
        }
        __asm__ volatile ("pause");
    }
    elapsed = read_tsc() - start;
    outb(SPKR, gate);
    if (elapsed == 0) {
        return 0;
    }
    return (elapsed * 1000u) / WINDOW_MS;
}

/* ----------------------------------------------------------------- data -- */
#define L1_ROWS 128u
#define L1_COLS 784u
#define L2_ROWS 10u
#define L2_COLS 128u

static float w1[L1_ROWS * L1_COLS] __attribute__((aligned(32)));
static float x1[L1_COLS]           __attribute__((aligned(32)));
static float b1[L1_ROWS];
static float o1s[L1_ROWS];
static float o1a[L1_ROWS];
static float o1f[L1_ROWS];

static float w2[L2_ROWS * L2_COLS] __attribute__((aligned(32)));
static float x2[L2_COLS]           __attribute__((aligned(32)));
static float b2[L2_ROWS];
static float o2s[L2_ROWS];
static float o2a[L2_ROWS];
static float o2f[L2_ROWS];

static void fill(void)
{
    uint32_t state = 0x12345u;
    uint32_t i;

    for (i = 0; i < L1_ROWS * L1_COLS; ++i) {
        state = state * 1664525u + 1013904223u;
        w1[i] = (float)((state >> 8) & 0xFFFFu) / 65535.0f - 0.4f;
    }
    for (i = 0; i < L1_COLS; ++i) {
        state = state * 1664525u + 1013904223u;
        x1[i] = (float)((state >> 8) & 0xFFFFu) / 65535.0f;
    }
    for (i = 0; i < L1_ROWS; ++i) {
        state = state * 1664525u + 1013904223u;
        b1[i] = (float)((state >> 8) & 0xFFFFu) / 65535.0f - 0.5f;
    }
    for (i = 0; i < L2_ROWS * L2_COLS; ++i) {
        state = state * 1664525u + 1013904223u;
        w2[i] = (float)((state >> 8) & 0xFFFFu) / 65535.0f - 0.4f;
    }
    for (i = 0; i < L2_COLS; ++i) {
        state = state * 1664525u + 1013904223u;
        x2[i] = (float)((state >> 8) & 0xFFFFu) / 65535.0f;
    }
    for (i = 0; i < L2_ROWS; ++i) {
        state = state * 1664525u + 1013904223u;
        b2[i] = (float)((state >> 8) & 0xFFFFu) / 65535.0f - 0.5f;
    }
}

/* Tuned scalar baseline, the same four-accumulator loop as src/ai.c. */
static void matvec_scalar(const float *w, const float *x, const float *b,
                          float *out, uint32_t rows, uint32_t cols)
{
    uint32_t r;
    uint32_t c;

    for (r = 0; r < rows; ++r) {
        const float *row = &w[r * cols];
        float s0 = b[r];
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;

        for (c = 0; c + 3u < cols; c += 4u) {
            s0 += row[c] * x[c];
            s1 += row[c + 1u] * x[c + 1u];
            s2 += row[c + 2u] * x[c + 2u];
            s3 += row[c + 3u] * x[c + 3u];
        }
        for (; c < cols; ++c) {
            s0 += row[c] * x[c];
        }
        out[r] = (s0 + s1) + (s2 + s3);
    }
}

static float absf(float value)
{
    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    bits.u &= 0x7FFFFFFFu;
    return bits.f;
}

/* Max absolute element-wise difference between two result vectors. */
static uint32_t max_diff(const float *reference, const float *candidate,
                         uint32_t count)
{
    float worst = 0.0f;
    uint32_t i;

    for (i = 0; i < count; ++i) {
        float d = absf(reference[i] - candidate[i]);

        if (d > worst) {
            worst = d;
        }
    }
    /* Single conversion; small enough to stay inside cvttss2si range. */
    return (uint32_t)(worst * 1000000.0f);
}

/* ---------------------------------------------------------------- timing -- */
#define WARMUP 64u
#define SAMPLES 8u

/* Returns the best-of-SAMPLES cycles per single matvec call. */
static uint64_t time_kernel(matvec_fn fn, const float *w, const float *x,
                            const float *b, float *out, uint32_t rows,
                            uint32_t cols, uint32_t reps)
{
    uint64_t best = UINT64_MAX;
    uint32_t s;
    uint32_t i;

    for (i = 0; i < WARMUP; ++i) {
        fn(w, x, b, out, rows, cols);
    }
    for (s = 0; s < SAMPLES; ++s) {
        uint64_t start = read_tsc();
        uint64_t delta;

        for (i = 0; i < reps; ++i) {
            fn(w, x, b, out, rows, cols);
        }
        delta = (read_tsc() - start) / reps;
        if (delta < best) {
            best = delta;
        }
    }
    return best;
}

/* Best-of-SAMPLES timing helper: the ratio to the scalar baseline is printed
 * as a fixed-point "N.dd", scaled by this constant. */
#define SCALED_ONE 100u

static void report_time(const char *label, uint64_t cycles,
                       uint64_t cycles_per_second, uint64_t scalar_cycles)
{
    puts_line("  ");
    puts_line(label);
    puts_line(" cycles=");
    putu(cycles);
    if (cycles_per_second != 0) {
        /* whole microseconds with two decimals, all in 64-bit integer math */
        uint64_t picos = cycles * 1000000000000ULL / cycles_per_second;
        uint64_t micros = picos / 1000000ULL;

        puts_line(" us=");
        putu(micros / 100u);
        putc_raw('.');
        putu(micros % 100u);
    }
    if (scalar_cycles != 0) {
        /* ratio to the scalar baseline, printed as "N.dd" via fixed-point */
        uint32_t scaled =
            (uint32_t)((uint64_t)SCALED_ONE * cycles / scalar_cycles);

        puts_line(" ratio=");
        putu(scaled / 100u);
        putc_raw('.');
        /* two-digit fraction with leading zero, else 3.06 prints as "3.6" */
        putc_raw((char)('0' + (scaled % 100u) / 10u));
        putc_raw((char)('0' + scaled % 10u));
    }
    puts_line("\n");
}

/* ------------------------------------------------------------------- main -- */
void kernel_main(uint32_t multiboot_info)
{
    uint32_t regs1[4];
    uint32_t regs7[4];
    uint64_t xcr0 = 0;
    uint64_t cycles_per_second;
    int have_avx;
    int have_avx2;
    int have_fma;
    int avx_state_ok;
    uint32_t l1_scalar;
    uint32_t l1_avx2 = 0;
    uint32_t l1_fma = 0;
    uint32_t l2_scalar;

    (void)multiboot_info;

    serial_init();
    puts_line("\n=== AVX2/FMA dispatch probe ===\n");

    cpuid_read(1u, 0u, regs1);
    cpuid_read(7u, 0u, regs7);
    have_avx = (regs1[2] & (1u << 28)) != 0;        /* ECX bit 28 */
    have_avx2 = (regs7[1] & (1u << 5)) != 0;        /* EBX bit 5   */
    have_fma = (regs1[2] & (1u << 12)) != 0;        /* ECX bit 12 */
    print_brand();

    puts_line("PROBE BEGIN\n");
    puts_line("  sse2=1\n");  /* x86-64 baseline: always present */
    puts_line("  avx=");
    puts_line(have_avx ? "1\n" : "0\n");
    puts_line("  avx2=");
    puts_line(have_avx2 ? "1\n" : "0\n");
    puts_line("  fma=");
    puts_line(have_fma ? "1\n" : "0\n");

    avx_state_ok = enable_avx_state(&xcr0);
    puts_line("  xcr0=");
    puthex64(xcr0);
    if (!avx_state_ok && (have_avx || have_avx2)) {
        puts_line("  (AVX state could not be enabled; SIMD kernels skipped)\n");
    }

    cycles_per_second = tsc_calibrate();
    puts_line("  cycles_per_second=");
    putu(cycles_per_second);
    puts_line("\n");

    fill();
    matvec_scalar(w1, x1, b1, o1s, L1_ROWS, L1_COLS);
    matvec_scalar(w2, x2, b2, o2s, L2_ROWS, L2_COLS);

    l1_scalar = (uint32_t)time_kernel(matvec_scalar, w1, x1, b1, o1s,
                                      L1_ROWS, L1_COLS, 256u);
    l2_scalar = (uint32_t)time_kernel(matvec_scalar, w2, x2, b2, o2s,
                                      L2_ROWS, L2_COLS, 4096u);
    puts_line("  layer1 128x784 scalar cycles=");
    putu(l1_scalar);
    puts_line("\n");
    puts_line("  layer2 10x128 scalar cycles=");
    putu(l2_scalar);
    puts_line("\n");

    if (have_avx && have_avx2 && avx_state_ok) {
        puts_line("  -- avx2 kernel executing --\n");
        matvec_avx2(w1, x1, b1, o1a, L1_ROWS, L1_COLS);
        matvec_avx2(w2, x2, b2, o2a, L2_ROWS, L2_COLS);
        puts_line("  layer1 maxdiff_scalar_avx2_scaled=");
        put_scaled(max_diff(o1s, o1a, L1_ROWS), 6u);
        puts_line("\n");
        puts_line("  layer2 maxdiff_scalar_avx2_scaled=");
        put_scaled(max_diff(o2s, o2a, L2_ROWS), 6u);
        puts_line("\n");
        l1_avx2 = (uint32_t)time_kernel(matvec_avx2, w1, x1, b1, o1a,
                                        L1_ROWS, L1_COLS, 256u);
        report_time("layer1 avx2", l1_avx2, cycles_per_second, l1_scalar);
        {
            uint32_t l2_avx2 = (uint32_t)time_kernel(matvec_avx2, w2, x2,
                                                     b2, o2a, L2_ROWS,
                                                     L2_COLS, 4096u);

            report_time("layer2 avx2", l2_avx2, cycles_per_second, l2_scalar);
        }
    }
    if (have_avx && have_avx2 && have_fma && avx_state_ok) {
        puts_line("  -- avx2+fma kernel executing --\n");
        matvec_fma(w1, x1, b1, o1f, L1_ROWS, L1_COLS);
        matvec_fma(w2, x2, b2, o2f, L2_ROWS, L2_COLS);
        puts_line("  layer1 maxdiff_scalar_fma_scaled=");
        put_scaled(max_diff(o1s, o1f, L1_ROWS), 6u);
        puts_line("\n");
        l1_fma = (uint32_t)time_kernel(matvec_fma, w1, x1, b1, o1f,
                                       L1_ROWS, L1_COLS, 256u);
        report_time("layer1 fma", l1_fma, cycles_per_second, l1_scalar);
        {
            uint32_t l2_fma = (uint32_t)time_kernel(matvec_fma, w2, x2,
                                                   b2, o2f, L2_ROWS,
                                                   L2_COLS, 4096u);

            report_time("layer2 fma", l2_fma, cycles_per_second, l2_scalar);
        }
    }

    report_time("layer1 scalar", l1_scalar, cycles_per_second, 0);
    report_time("layer2 scalar", l2_scalar, cycles_per_second, 0);
    puts_line("PROBE END\n");

    /* QEMU isa-debug-exit: same convention as the shipped kernel. */
    outb(0xF4u, 0x10u);
    for (;;) {
        __asm__ volatile ("cli; hlt");
    }
}
