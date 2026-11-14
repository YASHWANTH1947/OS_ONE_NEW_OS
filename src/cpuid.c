#include "ai.h"
#include "kernel.h"

/*
 * CPU feature detection for the unikernel (AEGIS-X System B, environment
 * layer). One snapshot is taken at boot; the dispatcher treats it as a set of
 * prerequisites, never as a decision.
 *
 * The enablement sequence was validated empirically by the probe:
 *
 *   1. CPUID.1.ECX[26] (XSAVE) and ECX[28] (AVX) are the STATIC capability
 *      gates. ECX[27] (OSXSAVE) is DYNAMIC - it mirrors CR4.OSXSAVE and reads
 *      0 until the kernel sets it, so it must not gate the check.
 *   2. CR4.OSXSAVE (bit 18) must be written before XSETBV is legal.
 *   3. XCR0 bits 0..2 (x87, SSE, AVX) must all be enabled; without YMM state
 *      enabled, the first VEX instruction raises #UD/#GP even when CPUID
 *      advertises AVX2.
 *   4. Only then may CPUID.1.ECX[27] be trusted as a confirmation.
 */

void ai_raw_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    __asm__ volatile ("cpuid"
                      : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                      : "a"(leaf), "c"(subleaf));
}

static int try_enable_avx_state(void)
{
    uint32_t regs[4];
    uint64_t cr4;
    uint32_t low;
    uint32_t high;

    ai_raw_cpuid(1u, 0u, regs);
    if ((regs[2] & (1u << 26)) == 0 || (regs[2] & (1u << 28)) == 0) {
        return 0;
    }

    __asm__ volatile ("movq %%cr4, %0" : "=r"(cr4));
    cr4 |= 0x40000ULL;                          /* CR4.OSXSAVE, bit 18 */
    __asm__ volatile ("movq %0, %%cr4" : : "r"(cr4) : "memory");

    /* XCR0 = bit0 x87 | bit1 SSE | bit2 AVX, all required for YMM use. */
    __asm__ volatile ("xsetbv" : : "a"(0x00000007u), "d"(0u), "c"(0u)
                      : "memory");
    __asm__ volatile ("xgetbv" : "=a"(low), "=d"(high) : "c"(0u));
    return (((uint64_t)high << 32) | low) == 0x7u;
}

int ai_probe_caps(ai_caps_t *caps)
{
    uint32_t regs1[4];
    uint32_t regs7[4];
    uint32_t max_extended;
    uint32_t line;

    if (caps == 0) {
        return 0;
    }
    caps->has_sse2 = 1;                   /* x86-64 baseline */
    caps->has_avx = 0;
    caps->has_avx2 = 0;
    caps->has_fma = 0;
    caps->avx_state_ready = 0;
    for (line = 0; line < 49u; ++line) {
        caps->brand[line] = '\0';
    }

    ai_raw_cpuid(0x80000000u, 0u, regs1);
    max_extended = regs1[0];
    if (max_extended >= 0x80000004u) {
        for (line = 0; line < 3u; ++line) {
            uint32_t part[4];

            ai_raw_cpuid(0x80000002u + line, 0u, part);
            ((uint32_t *)caps->brand)[line * 4u + 0u] = part[0];
            ((uint32_t *)caps->brand)[line * 4u + 1u] = part[1];
            ((uint32_t *)caps->brand)[line * 4u + 2u] = part[2];
            ((uint32_t *)caps->brand)[line * 4u + 3u] = part[3];
        }
        caps->brand[48] = '\0';
    }

    ai_raw_cpuid(1u, 0u, regs1);
    caps->has_avx = (regs1[2] & (1u << 28)) != 0;
    caps->has_fma = (regs1[2] & (1u << 12)) != 0;

    ai_raw_cpuid(7u, 0u, regs7);
    caps->has_avx2 = (regs7[1] & (1u << 5)) != 0;

    if (caps->has_avx) {
        caps->avx_state_ready = try_enable_avx_state();
    }
    return caps->avx_state_ready;
}
