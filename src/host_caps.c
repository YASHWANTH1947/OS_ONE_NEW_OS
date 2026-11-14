#define _POSIX_C_SOURCE 200809L

#include "ai.h"

/*
 * Host-side environment layer: CPU feature snapshot via CPUID, identical
 * sequence to the unikernel's src/cpuid.c. Under a hosted OS the AVX state
 * machine is already enabled by the OS, so probing is read-only; the
 * dynamic-OSXSAVE ordering caveat from the unikernel does not apply here.
 *
 * This TU is shared by every hosted build (host_bench, tests) so there is
 * exactly one definition of ai_probe_caps on the host side.
 */

void ai_raw_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    __asm__ volatile ("cpuid"
                      : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                      : "a"(leaf), "c"(subleaf));
}

int ai_probe_caps(ai_caps_t *caps)
{
    uint32_t regs1[4];
    uint32_t regs7[4];
    uint32_t extended_max;
    uint32_t line;

    if (caps == 0) {
        return 0;
    }
    caps->has_sse2 = 1;
    caps->has_avx = 0;
    caps->has_avx2 = 0;
    caps->has_fma = 0;
    caps->avx_state_ready = 0;
    for (line = 0; line < 49u; ++line) {
        caps->brand[line] = '\0';
    }

    ai_raw_cpuid(0x80000000u, 0u, regs1);
    extended_max = regs1[0];
    if (extended_max >= 0x80000004u) {
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

    /* Hosted: the OS has enabled the extended-state machine; the dynamic
     * OSXSAVE bit (ECX[27]) can be trusted directly as confirmation. */
    caps->avx_state_ready = caps->has_avx && (regs1[2] & (1u << 27)) != 0;
    return caps->avx_state_ready;
}
