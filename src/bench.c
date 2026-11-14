#include "bench.h"
#include "kernel.h"

/*
 * Deterministic benchmark harness:
 *   1. warm the instruction and data caches,
 *   2. time every forward pass individually with the timestamp counter,
 *   3. report percentiles rather than just a mean, because the tail is what a
 *      real-time workload actually has to budget for,
 *   4. score the same frozen MNIST samples the host binary scores.
 *
 * No allocation happens here; all storage is static and sized at compile time.
 */

static uint32_t latency_samples[BENCH_MAX_PASSES];
static uint32_t histogram[BENCH_HISTOGRAM_BUCKETS];

static void swap_u32(uint32_t *left, uint32_t *right)
{
    uint32_t temporary = *left;

    *left = *right;
    *right = temporary;
}

static void insertion_sort(uint32_t *values, uint32_t count)
{
    uint32_t i;

    for (i = 1; i < count; ++i) {
        uint32_t value = values[i];
        uint32_t position = i;

        while (position > 0 && values[position - 1] > value) {
            values[position] = values[position - 1];
            --position;
        }
        values[position] = value;
    }
}

static uint32_t partition(uint32_t *values, uint32_t count)
{
    uint32_t pivot = values[count - 1];
    uint32_t store = 0;
    uint32_t i;

    for (i = 0; i < count - 1u; ++i) {
        if (values[i] < pivot) {
            swap_u32(&values[i], &values[store]);
            ++store;
        }
    }
    swap_u32(&values[store], &values[count - 1u]);
    return store;
}

/* Quicksort with a median-of-three pivot; recursion only into the smaller
 * side keeps stack depth logarithmic even on adversarial input. */
static void sort_u32(uint32_t *values, uint32_t count)
{
    while (count > 16u) {
        uint32_t middle = count / 2u;
        uint32_t pivot_index;

        if (values[0] > values[middle]) {
            swap_u32(&values[0], &values[middle]);
        }
        if (values[0] > values[count - 1u]) {
            swap_u32(&values[0], &values[count - 1u]);
        }
        if (values[middle] > values[count - 1u]) {
            swap_u32(&values[middle], &values[count - 1u]);
        }
        swap_u32(&values[middle], &values[count - 1u]);

        pivot_index = partition(values, count);
        if (pivot_index < count - 1u - pivot_index) {
            sort_u32(values, pivot_index);
            values += pivot_index + 1u;
            count = count - pivot_index - 1u;
        } else {
            sort_u32(values + pivot_index + 1u, count - pivot_index - 1u);
            count = pivot_index;
        }
    }
    insertion_sort(values, count);
}

/* Nearest-rank percentile over an ascending array. */
static uint64_t percentile(const uint32_t *sorted, uint32_t count,
                           uint32_t permille)
{
    uint64_t rank = ((uint64_t)count * permille + 999u) / 1000u;

    if (rank < 1u) {
        rank = 1u;
    }
    if (rank > count) {
        rank = count;
    }
    return sorted[rank - 1u];
}

static uint64_t cycles_to_ns(uint64_t cycles, uint64_t cycles_per_second)
{
    uint64_t seconds;
    uint64_t remainder;
    uint64_t whole_ns;
    uint64_t fractional_ns;

    if (cycles_per_second == 0) {
        return 0;
    }
    seconds = cycles / cycles_per_second;
    if (seconds > UINT64_MAX / 1000000000ULL) {
        return UINT64_MAX;
    }
    whole_ns = seconds * 1000000000ULL;
    remainder = cycles % cycles_per_second;

    /* Calibration caps the frequency at 100 GHz, so this product fits in
     * 64 bits and retains the 10 ns precision displayed in the report. */
    fractional_ns =
        ((remainder * 100000000ULL) / cycles_per_second) * 10ULL;
    if (whole_ns > UINT64_MAX - fractional_ns) {
        return UINT64_MAX;
    }
    return whole_ns + fractional_ns;
}

static void build_histogram(uint32_t count, uint64_t min_cycles,
                            uint64_t max_cycles)
{
    uint64_t span = max_cycles - min_cycles;
    uint32_t i;

    for (i = 0; i < BENCH_HISTOGRAM_BUCKETS; ++i) {
        histogram[i] = 0;
    }
    if (span == 0) {
        histogram[0] = count;
        return;
    }
    for (i = 0; i < count; ++i) {
        uint64_t offset = (uint64_t)latency_samples[i] - min_cycles;
        uint32_t bucket = (uint32_t)((offset * BENCH_HISTOGRAM_BUCKETS) /
                                     (span + 1u));

        if (bucket >= BENCH_HISTOGRAM_BUCKETS) {
            bucket = BENCH_HISTOGRAM_BUCKETS - 1u;
        }
        ++histogram[bucket];
    }
}

void bench_measure(const ai_model_t *model, const float *input, uint32_t passes,
                   bench_latency_t *result)
{
    bench_measure_kernel(model, input, passes, AI_KERNEL_SCALAR, result);
}

void bench_measure_kernel(const ai_model_t *model, const float *input,
                          uint32_t passes, ai_kernel_id_t kernel,
                          bench_latency_t *result)
{
    float output[AI_OUTPUTS];
    uint64_t total = 0;
    uint64_t maximum = 0;
    uint32_t i;

    if (passes > BENCH_MAX_PASSES) {
        passes = BENCH_MAX_PASSES;
    }
    if (passes == 0) {
        passes = 1;
    }

    for (i = 0; i < BENCH_WARMUP_PASSES; ++i) {
        ai_infer_kernel(model, input, output, kernel);
    }

    for (i = 0; i < passes; ++i) {
        uint64_t start = read_tsc();
        uint64_t delta;

        ai_infer_kernel(model, input, output, kernel);
        delta = read_tsc() - start;
        if (delta > 0xFFFFFFFFu) {
            delta = 0xFFFFFFFFu;
        }
        latency_samples[i] = (uint32_t)delta;
        total += delta;
        if (delta > maximum) {
            maximum = delta;
        }
    }

    result->passes = passes;
    result->total_cycles = total;
    result->mean_cycles = total / passes;
    result->max_cycles = maximum;

    sort_u32(latency_samples, passes);
    result->min_cycles = latency_samples[0];
    result->p50_cycles = percentile(latency_samples, passes, 500u);
    result->p90_cycles = percentile(latency_samples, passes, 900u);
    result->p99_cycles = percentile(latency_samples, passes, 990u);
    result->max_cycles = latency_samples[passes - 1u];

    build_histogram(passes, result->min_cycles, result->max_cycles);
}

uint32_t bench_score(const ai_model_t *model, const ai_sample_t *samples,
                     uint32_t count)
{
    float output[AI_OUTPUTS];
    uint32_t correct = 0;
    uint32_t i;

    for (i = 0; i < count; ++i) {
        ai_infer_kernel(model, samples[i].pixels, output,
                        AI_KERNEL_SCALAR);
        if (ai_argmax(output) == samples[i].label) {
            ++correct;
        }
    }
    return correct;
}

uint32_t bench_score_kernel(const ai_model_t *model,
                            const ai_sample_t *samples, uint32_t count,
                            ai_kernel_id_t kernel)
{
    float output[AI_OUTPUTS];
    uint32_t correct = 0;
    uint32_t i;

    for (i = 0; i < count; ++i) {
        ai_infer_kernel(model, samples[i].pixels, output, kernel);
        if (ai_argmax(output) == samples[i].label) {
            ++correct;
        }
    }
    return correct;
}

static void print_microseconds(uint64_t nanoseconds)
{
    serial_printf("%llu.%02llu us", (unsigned long long)(nanoseconds / 1000u),
                  (unsigned long long)((nanoseconds % 1000u) / 10u));
}

static void print_percentage(uint32_t numerator, uint32_t denominator)
{
    uint64_t scaled = denominator == 0 ? 0 :
        ((uint64_t)numerator * 10000u) / denominator;

    serial_printf("%llu.%02llu%%", (unsigned long long)(scaled / 100u),
                  (unsigned long long)(scaled % 100u));
}

static void print_latency_line(const char *label, uint64_t cycles,
                               uint64_t cycles_per_second)
{
    serial_write(label);
    serial_repeat(' ', 24u - (uint32_t)serial_length(label));
    if (cycles_per_second != 0) {
        print_microseconds(cycles_to_ns(cycles, cycles_per_second));
    } else {
        serial_printf("%llu cycles", (unsigned long long)cycles);
    }
    serial_printf("  (%llu cycles)\n", (unsigned long long)cycles);
}

void bench_run(const ai_model_t *model, const ai_sample_t *samples,
               uint32_t sample_count, const bench_config_t *config,
               bench_result_t *result)
{
    float output[AI_OUTPUTS];
    uint64_t cycles_per_second = config->cycles_per_second;
    ai_kernel_id_t kernel = config->dispatch.selected;
    uint64_t total_ns;
    uint64_t throughput;
    uint64_t boot_to_inference_ns;
    uint32_t i;

    serial_write("\n");
    serial_rule();
    serial_write("  INFERENCE\n");
    serial_rule();

    ai_infer_kernel(model, samples[0].pixels, output, kernel);
    result->first_inference_cycles =
        read_tsc() - config->kernel_entry_cycles;
    for (i = 1; i < BENCH_WARMUP_PASSES; ++i) {
        ai_infer_kernel(model, samples[0].pixels, output, kernel);
    }
    result->samples = sample_count;
    result->correct = bench_score_kernel(model, samples, sample_count, kernel);

    bench_measure_kernel(model, samples[0].pixels, config->passes, kernel,
                         &result->latency);

    /* Consume every prediction so no forward pass can be elided. */
    result->checksum = 0;
    for (i = 0; i < 64u; ++i) {
        ai_infer_kernel(model, samples[i % sample_count].pixels,
                        output, kernel);
        result->checksum += ai_argmax(output);
    }

    total_ns = cycles_to_ns(result->latency.total_cycles, cycles_per_second);
    throughput = total_ns == 0 ? 0 :
        ((uint64_t)result->latency.passes * 1000000000ULL) / total_ns;
    boot_to_inference_ns =
        cycles_to_ns(result->first_inference_cycles, cycles_per_second);

    serial_write("  Accuracy ");
    serial_repeat('.', 26u);
    serial_printf(" %u / %u  (", result->correct, sample_count);
    print_percentage(result->correct, sample_count);
    serial_write(")\n");

    print_latency_line("  p50 latency", result->latency.p50_cycles,
                       cycles_per_second);
    print_latency_line("  p90 latency", result->latency.p90_cycles,
                       cycles_per_second);
    print_latency_line("  p99 latency", result->latency.p99_cycles,
                       cycles_per_second);
    print_latency_line("  min latency", result->latency.min_cycles,
                       cycles_per_second);
    print_latency_line("  max latency", result->latency.max_cycles,
                       cycles_per_second);

    serial_write("  Throughput ");
    serial_repeat('.', 24u);
    serial_printf(" %llu inferences/s\n", (unsigned long long)throughput);

    serial_write("  Boot -> first inference complete ");
    serial_repeat('.', 13u);
    if (cycles_per_second != 0) {
        print_microseconds(boot_to_inference_ns);
        serial_write("\n");
    } else {
        serial_printf(" %llu cycles\n",
                      (unsigned long long)result->first_inference_cycles);
    }

    serial_write("\n  RESULT BLOCK\n");
    serial_write("  @@BENCH_BEGIN\n");
    serial_write("  variant=unikernel\n");
    serial_write("  engine=bare-metal x86-64 (ring 0, single address space)\n");
    serial_printf("  passes=%u\n", result->latency.passes);
    serial_printf("  samples=%u\n", sample_count);
    serial_printf("  correct=%u\n", result->correct);
    serial_printf("  boot_to_inference_cycles=%llu\n",
                  (unsigned long long)result->first_inference_cycles);
    serial_printf("  cycles_per_second=%llu\n",
                  (unsigned long long)cycles_per_second);
    serial_printf("  total_cycles=%llu\n",
                  (unsigned long long)result->latency.total_cycles);
    serial_printf("  min_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.min_cycles,
                                                   cycles_per_second));
    serial_printf("  p50_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.p50_cycles,
                                                   cycles_per_second));
    serial_printf("  p90_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.p90_cycles,
                                                   cycles_per_second));
    serial_printf("  p99_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.p99_cycles,
                                                   cycles_per_second));
    serial_printf("  max_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.max_cycles,
                                                   cycles_per_second));
    serial_printf("  mean_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.mean_cycles,
                                                   cycles_per_second));
    serial_printf("  throughput_per_second=%llu\n",
                  (unsigned long long)throughput);
    if (cycles_per_second != 0) {
        serial_printf("  boot_to_inference_ns=%llu\n",
                      (unsigned long long)boot_to_inference_ns);
    } else {
        serial_write("  boot_to_inference_ns=0\n");
    }
    serial_printf("  kernel_bytes=%u\n", config->kernel_bytes);
    serial_printf("  model_bytes=%u\n", config->model_bytes);
    serial_printf("  samples_bytes=%u\n", config->samples_bytes);
    serial_printf("  free_frames=%u\n", config->free_frames);
    serial_printf("  usable_bytes=%llu\n",
                  (unsigned long long)config->usable_bytes);
    serial_printf("  checksum=%llu\n", (unsigned long long)result->checksum);

    /* Dispatch evidence: the selected kernel plus everything the measured
     * policy saw. Machine-parseable so the report can show why the kernel
     * used this code path on this machine. */
    serial_printf("  dispatch_kernel=%s\n",
                  ai_kernel_name(config->dispatch.selected));
    serial_printf("  dispatch_eligible_mask=%u\n",
                  config->dispatch.eligible_mask);
    serial_printf("  dispatch_measured_mask=%u\n",
                  config->dispatch.measured_mask);
    serial_printf("  dispatch_banned_mask=%u\n",
                  config->dispatch.banned_mask);
    serial_printf("  cpuid_brand=%s\n", config->caps.brand);
    serial_printf("  cpuid_avx=%d\n", config->caps.has_avx);
    serial_printf("  cpuid_avx2=%d\n", config->caps.has_avx2);
    serial_printf("  cpuid_fma=%d\n", config->caps.has_fma);
    serial_printf("  cpuid_avx_state=%d\n", config->caps.avx_state_ready);
    for (i = 0; i < AI_KERNEL_COUNT; ++i) {
        if ((config->dispatch.measured_mask & (1u << i)) == 0) {
            continue;
        }
        serial_printf("  dispatch_cycles_%s=%llu\n",
                      ai_kernel_name((ai_kernel_id_t)i),
                      (unsigned long long)config->dispatch.cycles[i]);
        if (i != AI_KERNEL_SCALAR) {
            serial_printf("  dispatch_diff_%s=%u\n",
                          ai_kernel_name((ai_kernel_id_t)i),
                          config->dispatch.diff[i]);
        }
    }

    serial_printf("  hist_min_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.min_cycles,
                                                   cycles_per_second));
    serial_printf("  hist_max_ns=%llu\n",
                  (unsigned long long)cycles_to_ns(result->latency.max_cycles,
                                                   cycles_per_second));
    serial_write("  hist_ns=");
    for (i = 0; i < BENCH_HISTOGRAM_BUCKETS; ++i) {
        serial_printf("%s%u", i == 0 ? "" : ",", histogram[i]);
    }
    serial_write("\n  @@BENCH_END\n");
}
