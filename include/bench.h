#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

#include "ai.h"

#define BENCH_MAX_PASSES 20000u
#define BENCH_WARMUP_PASSES 32u
#define BENCH_HISTOGRAM_BUCKETS 16u

typedef struct {
    uint32_t passes;
    uint64_t total_cycles;
    uint64_t min_cycles;
    uint64_t p50_cycles;
    uint64_t p90_cycles;
    uint64_t p99_cycles;
    uint64_t max_cycles;
    uint64_t mean_cycles;
} bench_latency_t;

typedef struct {
    uint32_t passes;
    uint32_t kernel_bytes;
    uint32_t model_bytes;
    uint32_t samples_bytes;
    uint32_t free_frames;
    uint64_t usable_bytes;
    uint64_t cycles_per_second;
    uint64_t kernel_entry_cycles;
    ai_caps_t caps;
    ai_dispatch_record_t dispatch;
} bench_config_t;

typedef struct {
    bench_latency_t latency;
    uint32_t samples;
    uint32_t correct;
    uint64_t first_inference_cycles;
    uint64_t checksum;
} bench_result_t;

void bench_measure(const ai_model_t *model, const float *input, uint32_t passes,
                   bench_latency_t *result);
uint32_t bench_score(const ai_model_t *model, const ai_sample_t *samples,
                     uint32_t count);
/* Bench with an explicitly chosen kernel: every number the report shows
   names the code path that produced it. */
void bench_measure_kernel(const ai_model_t *model, const float *input,
                          uint32_t passes, ai_kernel_id_t kernel,
                          bench_latency_t *result);
uint32_t bench_score_kernel(const ai_model_t *model,
                            const ai_sample_t *samples, uint32_t count,
                            ai_kernel_id_t kernel);
void bench_run(const ai_model_t *model, const ai_sample_t *samples,
               uint32_t sample_count, const bench_config_t *config,
               bench_result_t *result);

#endif
