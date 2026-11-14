#include "kernel.h"

/*
 * Timestamp counter calibration using PIT channel 2 (the PC speaker timer,
 * which needs no interrupt vector and no scheduler). We program a one-shot
 * countdown of a known duration, poll the output bit, and divide the elapsed
 * TSC ticks by that duration. This turns raw cycle counts into real time.
 */

#define PIT_CHANNEL2 0x42u
#define PIT_COMMAND 0x43u
#define SPEAKER_PORT 0x61u
#define PIT_BASE_FREQUENCY 1193182u
#define CALIBRATION_WINDOW_MS 50u
#define CALIBRATION_POLL_LIMIT 5000000u

#define QEMU_DEBUG_EXIT_PORT 0xF4u
#define QEMU_DEBUG_EXIT_MAGIC 0x10u

uint64_t read_tsc(void)
{
    uint32_t low;
    uint32_t high;

    __asm__ volatile ("lfence; rdtsc; lfence"
                      : "=a"(low), "=d"(high) : : "memory");
    return ((uint64_t)high << 32) | low;
}

static void pit_program_countdown(uint32_t divisor)
{
    /* Channel 2, lobyte/hibyte access, mode 0 (interrupt on terminal count). */
    port_out(PIT_COMMAND, 0xB0u);
    port_out(PIT_CHANNEL2, (uint8_t)(divisor & 0xFFu));
    port_out(PIT_CHANNEL2, (uint8_t)((divisor >> 8) & 0xFFu));
}

uint64_t tsc_calibrate(void)
{
    uint32_t divisor = (uint32_t)(((uint64_t)PIT_BASE_FREQUENCY *
                                   CALIBRATION_WINDOW_MS) / 1000u);
    uint8_t speaker_state = (uint8_t)(port_in(SPEAKER_PORT) & 0xFCu);
    uint32_t polls = 0;
    uint64_t start;
    uint64_t elapsed;
    uint64_t cycles_per_second;

    if (divisor == 0) {
        return 0;
    }

    /* Gate low and speaker disconnected, then arm the countdown. */
    port_out(SPEAKER_PORT, speaker_state);
    pit_program_countdown(divisor);
    port_out(SPEAKER_PORT, (uint8_t)(speaker_state | 0x01u));

    start = read_tsc();
    while ((port_in(SPEAKER_PORT) & 0x20u) == 0) {
        if (++polls > CALIBRATION_POLL_LIMIT) {
            port_out(SPEAKER_PORT, speaker_state);
            return 0;
        }
        __asm__ volatile ("pause");
    }
    elapsed = read_tsc() - start;
    port_out(SPEAKER_PORT, speaker_state);

    if (elapsed == 0) {
        return 0;
    }
    cycles_per_second = (elapsed * 1000u) / CALIBRATION_WINDOW_MS;

    /* Reject implausible results rather than reporting fabricated timings. */
    if (cycles_per_second < 100000000ULL ||
        cycles_per_second > 100000000000ULL) {
        return 0;
    }
    return cycles_per_second;
}

/*
 * QEMU's isa-debug-exit device: writing 0x10 | status terminates the VM with
 * a deterministic exit code, which is what makes the benchmark automatable.
 * On real hardware the write is harmless and the halt loop below runs instead.
 */
void qemu_debug_exit(uint32_t code)
{
    port_out(QEMU_DEBUG_EXIT_PORT,
             (uint8_t)(QEMU_DEBUG_EXIT_MAGIC | (code & 0x0Fu)));
}
