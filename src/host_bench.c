#define _POSIX_C_SOURCE 200809L

#include "ai.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/utsname.h>

/*
 * Baseline environment: the same forward-pass kernels, compiled as an
 * ordinary Linux process. It consumes the same model bytes and the same frozen
 * samples as the unikernel, so accuracy is directly comparable and
 * latency numbers describe this host, not a synthetic loop.
 *
 * AEGIS-X factorial support: --kernel forces a code path (scalar | avx2 |
 * fma) so the E1/E2 Linux cells pair with the unikernel's E4/E5 cells under
 * one methodology; without --kernel, the same measured dispatch policy as
 * the unikernel runs and its selection is reported. --csv appends an
 * aggregatable row instead of (only) producing the full JSON.
 */

#define HOST_MAX_PASSES 20000u
#define HOST_WARMUP_PASSES 32u
#define HOST_HISTOGRAM_BUCKETS 16u

static uint64_t latency_ns[HOST_MAX_PASSES];
static uint32_t histogram[HOST_HISTOGRAM_BUCKETS];

/* Cycle source for the dispatch policy's timing loop. On the host, RDTSC is
 * only meaningful after a frequency estimate; conversions stay in
 * CLOCK_MONOTONIC nanoseconds, so cycles are used solely for RELATIVE
 * kernel comparison inside one run (which is all the policy needs). */
static uint64_t host_tsc(void)
{
    uint32_t lo;
    uint32_t hi;

    __asm__ volatile ("lfence; rdtsc; lfence" : "=a"(lo), "=d"(hi)
                      : : "memory");
    return ((uint64_t)hi << 32) | lo;
}

/* Host-side environment layer: src/host_caps.c (linked into every hosted
 * build) provides ai_probe_caps/ai_raw_cpuid, identical to the unikernel's
 * src/cpuid.c. Under a hosted OS the AVX state machine is already enabled
 * by the OS, so probing is read-only; host_bench only supplies the cycle
 * source for the dispatch policy. */

static int now_ns(uint64_t *result)
{
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    *result = (uint64_t)value.tv_sec * 1000000000ULL +
              (uint64_t)value.tv_nsec;
    return 1;
}

static int compare_u64(const void *left, const void *right)
{
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;

    return (a > b) - (a < b);
}

/* Nearest-rank percentile, identical definition to the unikernel. */
static uint64_t percentile(const uint64_t *sorted, uint32_t count,
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

static void build_histogram(uint32_t count, uint64_t min_value,
                            uint64_t max_value)
{
    uint64_t span = max_value - min_value;
    uint32_t i;

    for (i = 0; i < HOST_HISTOGRAM_BUCKETS; ++i) {
        histogram[i] = 0;
    }
    if (span == 0) {
        histogram[0] = count;
        return;
    }
    for (i = 0; i < count; ++i) {
        uint64_t offset = latency_ns[i] - min_value;
        uint32_t bucket =
            (uint32_t)((offset * HOST_HISTOGRAM_BUCKETS) / (span + 1u));

        if (bucket >= HOST_HISTOGRAM_BUCKETS) {
            bucket = HOST_HISTOGRAM_BUCKETS - 1u;
        }
        ++histogram[bucket];
    }
}

static void *read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    long length;
    void *buffer;
    size_t read_bytes;

    if (file == NULL) {
        perror("error: unable to open file");
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "error: unable to size %s\n", path);
        fclose(file);
        return NULL;
    }
    buffer = malloc((size_t)length);
    if (buffer == NULL) {
        fprintf(stderr, "error: out of memory for %s\n", path);
        fclose(file);
        return NULL;
    }
    read_bytes = fread(buffer, 1, (size_t)length, file);
    if (read_bytes != (size_t)length || fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "error: unable to read %s\n", path);
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = read_bytes;
    return buffer;
}

static void json_escape(const char *text, char *out, size_t capacity)
{
    size_t written = 0;

    while (*text != '\0' && written + 7u < capacity) {
        unsigned char character = (unsigned char)*text++;

        if (character == '"' || character == '\\') {
            out[written++] = '\\';
            out[written++] = (char)character;
        } else if (character < 0x20u) {
            written += (size_t)snprintf(out + written, capacity - written,
                                        "\\u%04x", character);
        } else {
            out[written++] = (char)character;
        }
    }
    out[written] = '\0';
}

static void read_cpu_model(char *out, size_t capacity)
{
    FILE *file = fopen("/proc/cpuinfo", "r");
    char line[512];

    out[0] = '\0';
    if (file == NULL) {
        return;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strncmp(line, "model name", 10) == 0) {
            const char *colon = strchr(line, ':');

            if (colon != NULL) {
                size_t length;

                colon += 1;
                while (*colon == ' ' || *colon == '\t') {
                    ++colon;
                }
                length = strcspn(colon, "\r\n");
                if (length >= capacity) {
                    length = capacity - 1u;
                }
                memcpy(out, colon, length);
                out[length] = '\0';
            }
            break;
        }
    }
    fclose(file);
}

static unsigned long long read_memory_bytes(void)
{
    FILE *file = fopen("/proc/meminfo", "r");
    char key[64];
    unsigned long long value;
    char unit[16];

    if (file == NULL) {
        return 0;
    }
    while (fscanf(file, "%63s %llu %15s", key, &value, unit) == 3) {
        if (strcmp(key, "MemTotal:") == 0) {
            fclose(file);
            return value * 1024ULL;
        }
    }
    fclose(file);
    return 0;
}

static unsigned long long read_process_rss_bytes(void)
{
    FILE *file = fopen("/proc/self/statm", "r");
    unsigned long long total_pages;
    unsigned long long resident_pages;
    long page_size;

    if (file == NULL) {
        return 0;
    }
    if (fscanf(file, "%llu %llu", &total_pages, &resident_pages) != 2) {
        fclose(file);
        return 0;
    }
    (void)total_pages;
    if (fclose(file) != 0) {
        return 0;
    }
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 ||
        resident_pages > ULLONG_MAX / (unsigned long long)page_size) {
        return 0;
    }
    return resident_pages * (unsigned long long)page_size;
}

static int running_in_container(void)
{
    FILE *file;

    if (access("/.dockerenv", F_OK) == 0) {
        return 1;
    }
    file = fopen("/proc/1/cgroup", "r");
    if (file != NULL) {
        char line[256];
        int found = 0;

        while (fgets(line, sizeof(line), file) != NULL) {
            if (strstr(line, "docker") != NULL ||
                strstr(line, "kubepods") != NULL ||
                strstr(line, "containerd") != NULL) {
                found = 1;
                break;
            }
        }
        fclose(file);
        return found;
    }
    return 0;
}

static void print_microseconds(const char *label, uint64_t nanoseconds)
{
    size_t pad = strlen(label) < 24u ? 24u - strlen(label) : 1u;

    printf("%s", label);
    while (pad-- > 0) {
        putchar(' ');
    }
    printf("%llu.%02llu us\n", (unsigned long long)(nanoseconds / 1000u),
           (unsigned long long)((nanoseconds % 1000u) / 10u));
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <model.bin> <samples.bin> [--passes N] [--json PATH]\n"
            "                     [--kernel scalar|avx2|fma] [--csv PATH]\n",
            program);
}

static int parse_kernel(const char *name, ai_kernel_id_t *kernel)
{
    if (name == NULL) {
        return 0;
    }
    if (strcmp(name, "scalar") == 0) {
        *kernel = AI_KERNEL_SCALAR;
        return 1;
    }
    if (strcmp(name, "avx2") == 0) {
        *kernel = AI_KERNEL_AVX2;
        return 1;
    }
    if (strcmp(name, "fma") == 0) {
        *kernel = AI_KERNEL_FMA;
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    ai_model_t *model;
    ai_sample_t *samples;
    void *samples_raw;
    size_t model_bytes = 0;
    size_t samples_bytes = 0;
    uint32_t sample_count = 0;
    uint32_t passes = 10000u;
    uint32_t correct = 0;
    uint32_t i;
    float output[AI_OUTPUTS];
    uint64_t total = 0;
    uint64_t min_ns;
    uint64_t p50_ns;
    uint64_t p90_ns;
    uint64_t p99_ns;
    uint64_t max_ns;
    uint64_t mean_ns;
    uint64_t throughput;
    uint64_t boot_to_inference_ns;
    uint64_t start;
    uint64_t setup_end;
    uint64_t first_inference_end;
    uint64_t pass_start;
    uint64_t pass_end;
    const char *json_path = NULL;
    char cpu_model[192];
    struct utsname system_info;
    char measured_at[64];
    time_t wall_clock = time(NULL);
    struct tm utc;
    int container;
    unsigned long long process_rss_bytes;
    ai_caps_t caps;
    ai_dispatch_record_t dispatch;
    ai_kernel_id_t kernel = AI_KERNEL_SCALAR;
    ai_kernel_id_t forced_kernel = AI_KERNEL_COUNT;
    int forced = 0;
    const char *csv_path = NULL;
    const char *kernel_label;

    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    for (i = 3; i < (uint32_t)argc; ++i) {
        if (strcmp(argv[i], "--json") == 0 && i + 1u < (uint32_t)argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 &&
                   i + 1u < (uint32_t)argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--passes") == 0 &&
                   i + 1u < (uint32_t)argc) {
            char *end;
            long parsed;

            errno = 0;
            parsed = strtol(argv[++i], &end, 10);

            if (errno != 0 || *end != '\0' || parsed < 1 ||
                parsed > (long)HOST_MAX_PASSES) {
                fprintf(stderr, "error: --passes must be 1..%u\n",
                        HOST_MAX_PASSES);
                return 2;
            }
            passes = (uint32_t)parsed;
        } else if (strcmp(argv[i], "--kernel") == 0 &&
                   i + 1u < (uint32_t)argc) {
            if (!parse_kernel(argv[++i], &forced_kernel)) {
                fprintf(stderr,
                        "error: --kernel must be scalar, avx2, or fma\n");
                return 2;
            }
            forced = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!now_ns(&start)) {
        perror("error: CLOCK_MONOTONIC");
        return 1;
    }
    model = (ai_model_t *)read_file(argv[1], &model_bytes);
    if (model == NULL) {
        return 1;
    }
    samples_raw = read_file(argv[2], &samples_bytes);
    if (samples_raw == NULL) {
        free(model);
        return 1;
    }
    samples = (ai_sample_t *)samples_raw;
    if (samples_bytes < sizeof(ai_sample_header_t)) {
        fputs("error: sample file is too small\n", stderr);
        free(samples_raw);
        free(model);
        return 1;
    }
    samples = (ai_sample_t *)((uint8_t *)samples_raw +
                              sizeof(ai_sample_header_t));
    if (!ai_samples_valid(samples_raw, samples_bytes, &sample_count) ||
        sample_count > HOST_MAX_PASSES) {
        fputs("error: sample file has an invalid format or size\n", stderr);
        free(samples_raw);
        free(model);
        return 1;
    }
    if (model_bytes != sizeof(ai_model_t) ||
        !ai_model_valid(model, model_bytes)) {
        fputs("error: model has an invalid header or size\n", stderr);
        free(samples_raw);
        free(model);
        return 1;
    }
    if (!now_ns(&setup_end)) {
        perror("error: CLOCK_MONOTONIC");
        free(samples_raw);
        free(model);
        return 1;
    }

    /* Environment snapshot and kernel selection. Forced kernels must still
     * be eligible; a forced-but-ineligible kernel is an error, not a silent
     * substitution, so an experiment cell never lies about its code path. */
    ai_probe_caps(&caps);
    if (forced) {
        if (!ai_kernel_eligible(forced_kernel, &caps)) {
            fprintf(stderr, "error: --kernel %s is not available on this "
                            "CPU (avx2=%d fma=%d state=%d)\n",
                    ai_kernel_name(forced_kernel), caps.has_avx2,
                    caps.has_fma, caps.avx_state_ready);
            free(samples_raw);
            free(model);
            return 1;
        }
        kernel = forced_kernel;
        memset(&dispatch, 0, sizeof(dispatch));
        dispatch.selected = kernel;
        dispatch.eligible_mask = 1u << kernel;
        dispatch.measured_mask = 1u << kernel;
    } else {
        kernel = ai_dispatch(model, ai_dispatch_input(), &caps, host_tsc,
                             &dispatch);
    }
    kernel_label = ai_kernel_name(kernel);

    ai_infer_kernel(model, samples[0].pixels, output, kernel);
    if (!now_ns(&first_inference_end)) {
        perror("error: CLOCK_MONOTONIC");
        free(samples_raw);
        free(model);
        return 1;
    }
    if (first_inference_end < start) {
        fputs("error: CLOCK_MONOTONIC moved backwards\n", stderr);
        free(samples_raw);
        free(model);
        return 1;
    }
    boot_to_inference_ns = first_inference_end - start;

    for (i = 1; i < HOST_WARMUP_PASSES; ++i) {
        ai_infer_kernel(model, samples[0].pixels, output, kernel);
    }

    for (i = 0; i < sample_count; ++i) {
        ai_infer_kernel(model, samples[i].pixels, output, kernel);
        if (ai_argmax(output) == samples[i].label) {
            ++correct;
        }
    }

    for (i = 0; i < passes; ++i) {
        uint64_t delta;

        if (!now_ns(&pass_start)) {
            perror("error: CLOCK_MONOTONIC");
            free(samples_raw);
            free(model);
            return 1;
        }
        ai_infer_kernel(model, samples[0].pixels, output, kernel);
        if (!now_ns(&pass_end)) {
            perror("error: CLOCK_MONOTONIC");
            free(samples_raw);
            free(model);
            return 1;
        }
        if (pass_end < pass_start) {
            fputs("error: CLOCK_MONOTONIC moved backwards\n", stderr);
            free(samples_raw);
            free(model);
            return 1;
        }
        delta = pass_end - pass_start;
        latency_ns[i] = delta;
        total += delta;
    }

    qsort(latency_ns, passes, sizeof(latency_ns[0]), compare_u64);
    min_ns = latency_ns[0];
    p50_ns = percentile(latency_ns, passes, 500u);
    p90_ns = percentile(latency_ns, passes, 900u);
    p99_ns = percentile(latency_ns, passes, 990u);
    max_ns = latency_ns[passes - 1u];
    mean_ns = total / passes;
    throughput = total == 0 ? 0 : ((uint64_t)passes * 1000000000ULL) / total;
    build_histogram(passes, min_ns, max_ns);

    read_cpu_model(cpu_model, sizeof(cpu_model));
    container = running_in_container();
    if (uname(&system_info) != 0) {
        perror("error: uname");
        free(samples_raw);
        free(model);
        return 1;
    }
    if (wall_clock == (time_t)-1 || gmtime_r(&wall_clock, &utc) == NULL ||
        strftime(measured_at, sizeof(measured_at), "%Y-%m-%dT%H:%M:%SZ",
                 &utc) == 0) {
        fputs("error: unable to produce benchmark timestamp\n", stderr);
        free(samples_raw);
        free(model);
        return 1;
    }
    process_rss_bytes = read_process_rss_bytes();

    printf("  BASELINE: native Linux process\n");
    printf("  ----------------------------------------------------------\n");
    printf("  Kernel ................ %s%s\n", kernel_label,
           forced ? " (forced)" : " (measured dispatch)");
    printf("  CPU .................. avx2=%d fma=%d state=%d\n",
           caps.has_avx2, caps.has_fma, caps.avx_state_ready);
    printf("  Accuracy ................ %u / %u  (%.2f%%)\n", correct,
           sample_count,
           sample_count == 0 ? 0.0 :
               100.0 * (double)correct / (double)sample_count);
    print_microseconds("  p50 latency", p50_ns);
    print_microseconds("  p90 latency", p90_ns);
    print_microseconds("  p99 latency", p99_ns);
    print_microseconds("  min latency", min_ns);
    print_microseconds("  max latency", max_ns);
    printf("  Throughput .............. %llu inferences/s\n",
           (unsigned long long)throughput);
    printf("  Process RSS ............. %llu bytes\n", process_rss_bytes);
    printf("  Samples loaded in ....... %.2f ms\n",
           (double)(setup_end - start) / 1e6);

    printf("\n  @@BENCH_BEGIN\n");
    printf("  variant=linux-host\n");
    printf("  engine=native userspace process\n");
    printf("  kernel=%s\n", kernel_label);
    printf("  kernel_forced=%d\n", forced);
    printf("  dispatch_kernel=%s\n", kernel_label);
    printf("  dispatch_eligible_mask=%u\n", dispatch.eligible_mask);
    printf("  dispatch_measured_mask=%u\n", dispatch.measured_mask);
    printf("  dispatch_banned_mask=%u\n", dispatch.banned_mask);
    printf("  cpuid_brand=%s\n", caps.brand);
    printf("  cpuid_avx=%d\n", caps.has_avx);
    printf("  cpuid_avx2=%d\n", caps.has_avx2);
    printf("  cpuid_fma=%d\n", caps.has_fma);
    printf("  cpuid_avx_state=%d\n", caps.avx_state_ready);
    for (i = 0; i < (uint32_t)AI_KERNEL_COUNT; ++i) {
        if ((dispatch.measured_mask & (1u << i)) == 0) {
            continue;
        }
        printf("  dispatch_cycles_%s=%llu\n",
               ai_kernel_name((ai_kernel_id_t)i),
               (unsigned long long)dispatch.cycles[i]);
        if (i != (uint32_t)AI_KERNEL_SCALAR) {
            printf("  dispatch_diff_%s=%u\n",
                   ai_kernel_name((ai_kernel_id_t)i), dispatch.diff[i]);
        }
    }
    printf("  passes=%u\n", passes);
    printf("  samples=%u\n", sample_count);
    printf("  correct=%u\n", correct);
    printf("  min_ns=%llu\n", (unsigned long long)min_ns);
    printf("  p50_ns=%llu\n", (unsigned long long)p50_ns);
    printf("  p90_ns=%llu\n", (unsigned long long)p90_ns);
    printf("  p99_ns=%llu\n", (unsigned long long)p99_ns);
    printf("  max_ns=%llu\n", (unsigned long long)max_ns);
    printf("  mean_ns=%llu\n", (unsigned long long)mean_ns);
    printf("  throughput_per_second=%llu\n", (unsigned long long)throughput);
    printf("  boot_to_inference_ns=%llu\n",
           (unsigned long long)boot_to_inference_ns);
    printf("  kernel_bytes=0\n");
    printf("  model_bytes=%llu\n", (unsigned long long)model_bytes);
    printf("  samples_bytes=%llu\n", (unsigned long long)samples_bytes);
    printf("  hist_ns=");
    for (i = 0; i < HOST_HISTOGRAM_BUCKETS; ++i) {
        printf("%s%u", i == 0 ? "" : ",", histogram[i]);
    }
    printf("\n  hist_min_ns=%llu\n", (unsigned long long)min_ns);
    printf("  hist_max_ns=%llu\n", (unsigned long long)max_ns);
    printf("  elapsed_ns=%llu\n", (unsigned long long)total);
    printf("  @@BENCH_END\n");

    if (json_path != NULL) {
        FILE *json = fopen(json_path, "w");
        char escaped_cpu[512];
        char escaped_node[512];
        char escaped_release[512];

        if (json == NULL) {
            perror("error: unable to write json");
            free(samples_raw);
            free(model);
            return 1;
        }
        json_escape(cpu_model, escaped_cpu, sizeof(escaped_cpu));
        json_escape(system_info.nodename, escaped_node,
                    sizeof(escaped_node));
        json_escape(system_info.release, escaped_release,
                    sizeof(escaped_release));
        fprintf(json, "{\n");
        fprintf(json, "  \"variant\": \"linux-host\",\n");
        fprintf(json, "  \"engine\": \"native userspace process\",\n");
        fprintf(json, "  \"kernel\": \"%s\",\n", kernel_label);
        fprintf(json, "  \"kernel_forced\": %s,\n",
                forced ? "true" : "false");
        fprintf(json, "  \"dispatch\": {\n");
        fprintf(json, "    \"selected\": \"%s\",\n",
                ai_kernel_name(dispatch.selected));
        fprintf(json, "    \"eligible_mask\": %u,\n",
                dispatch.eligible_mask);
        fprintf(json, "    \"measured_mask\": %u,\n",
                dispatch.measured_mask);
        fprintf(json, "    \"banned_mask\": %u,\n", dispatch.banned_mask);
        fprintf(json, "    \"cycles\": {");
        {
            int first = 1;

            for (i = 0; i < (uint32_t)AI_KERNEL_COUNT; ++i) {
                if ((dispatch.measured_mask & (1u << i)) == 0) {
                    continue;
                }
                fprintf(json, "%s\"%s\": %llu", first ? "" : ", ",
                        ai_kernel_name((ai_kernel_id_t)i),
                        (unsigned long long)dispatch.cycles[i]);
                first = 0;
            }
        }
        fprintf(json, "},\n");
        fprintf(json, "    \"diff\": {");
        {
            int first = 1;

            for (i = 0; i < (uint32_t)AI_KERNEL_COUNT; ++i) {
                if ((dispatch.measured_mask & (1u << i)) == 0 ||
                    i == (uint32_t)AI_KERNEL_SCALAR) {
                    continue;
                }
                fprintf(json, "%s\"%s\": %u", first ? "" : ", ",
                        ai_kernel_name((ai_kernel_id_t)i), dispatch.diff[i]);
                first = 0;
            }
        }
        fprintf(json, "}\n");
        fprintf(json, "  },\n");
        fprintf(json, "  \"caps\": {\n");
        fprintf(json, "    \"brand\": \"%s\",\n", caps.brand);
        fprintf(json, "    \"avx\": %d,\n", caps.has_avx);
        fprintf(json, "    \"avx2\": %d,\n", caps.has_avx2);
        fprintf(json, "    \"fma\": %d,\n", caps.has_fma);
        fprintf(json, "    \"avx_state\": %d\n", caps.avx_state_ready);
        fprintf(json, "  },\n");
        fprintf(json, "  \"measured_at\": \"%s\",\n", measured_at);
        fprintf(json, "  \"passes\": %u,\n", passes);
        fprintf(json, "  \"samples\": %u,\n", sample_count);
        fprintf(json, "  \"correct\": %u,\n", correct);
        fprintf(json, "  \"min_ns\": %llu,\n", (unsigned long long)min_ns);
        fprintf(json, "  \"p50_ns\": %llu,\n", (unsigned long long)p50_ns);
        fprintf(json, "  \"p90_ns\": %llu,\n", (unsigned long long)p90_ns);
        fprintf(json, "  \"p99_ns\": %llu,\n", (unsigned long long)p99_ns);
        fprintf(json, "  \"max_ns\": %llu,\n", (unsigned long long)max_ns);
        fprintf(json, "  \"mean_ns\": %llu,\n", (unsigned long long)mean_ns);
        fprintf(json, "  \"throughput_per_second\": %llu,\n",
                (unsigned long long)throughput);
        fprintf(json, "  \"boot_to_inference_ns\": %llu,\n",
                (unsigned long long)boot_to_inference_ns);
        fprintf(json, "  \"kernel_bytes\": 0,\n");
        fprintf(json, "  \"process_rss_bytes\": %llu,\n",
                process_rss_bytes);
        fprintf(json, "  \"model_bytes\": %llu,\n",
                (unsigned long long)model_bytes);
        fprintf(json, "  \"samples_bytes\": %llu,\n",
                (unsigned long long)samples_bytes);
        fprintf(json, "  \"hist_min_ns\": %llu,\n",
                (unsigned long long)min_ns);
        fprintf(json, "  \"hist_max_ns\": %llu,\n",
                (unsigned long long)max_ns);
        fprintf(json, "  \"hist_ns\": [");
        for (i = 0; i < HOST_HISTOGRAM_BUCKETS; ++i) {
            fprintf(json, "%s%u", i == 0 ? "" : ", ", histogram[i]);
        }
        fprintf(json, "],\n");
        fprintf(json, "  \"host\": {\n");
        fprintf(json, "    \"cpu_model\": \"%s\",\n", escaped_cpu);
        fprintf(json, "    \"cores\": %ld,\n",
                sysconf(_SC_NPROCESSORS_ONLN));
        fprintf(json, "    \"memory_bytes\": %llu,\n", read_memory_bytes());
        fprintf(json, "    \"container\": %s,\n",
                container ? "true" : "false");
        fprintf(json, "    \"nodename\": \"%s\",\n", escaped_node);
        fprintf(json, "    \"kernel_release\": \"%s\"\n", escaped_release);
        fprintf(json, "  }\n");
        fprintf(json, "}\n");
        if (fclose(json) != 0) {
            perror("error: unable to close json");
            free(samples_raw);
            free(model);
            return 1;
        }
        printf("\n  Wrote %s\n", json_path);
    }

    if (csv_path != NULL) {
        /* One CSV row per run for factorial aggregation across cells.
         * Header is written only for a fresh file, so appending the
         * E1..En cells into one results.csv just works. */
        FILE *csv = fopen(csv_path, "r");
        int had_header = csv != NULL;
        char line[8];

        if (csv != NULL) {
            had_header = fgets(line, sizeof(line), csv) != NULL;
            if (fclose(csv) != 0) {
                perror("error: unable to close csv");
                free(samples_raw);
                free(model);
                return 1;
            }
        }
        csv = fopen(csv_path, "a");
        if (csv == NULL) {
            perror("error: unable to write csv");
            free(samples_raw);
            free(model);
            return 1;
        }
        if (!had_header) {
            fprintf(csv, "variant,kernel,forced,passes,samples,correct,"
                         "min_ns,p50_ns,p90_ns,p99_ns,max_ns,mean_ns,"
                         "throughput_per_second,boot_to_inference_ns,"
                         "dispatch_cycles_scalar,dispatch_cycles_avx2,"
                         "dispatch_cycles_fma,cpuid_avx2,cpuid_fma,"
                         "cpuid_avx_state,measured_at\n");
        }
        fprintf(csv, "linux-host,%s,%d,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,"
                    "%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%d,%s\n",
                kernel_label, forced, passes, sample_count, correct,
                (unsigned long long)min_ns, (unsigned long long)p50_ns,
                (unsigned long long)p90_ns, (unsigned long long)p99_ns,
                (unsigned long long)max_ns, (unsigned long long)mean_ns,
                (unsigned long long)throughput,
                (unsigned long long)boot_to_inference_ns,
                (unsigned long long)dispatch.cycles[AI_KERNEL_SCALAR],
                (unsigned long long)dispatch.cycles[AI_KERNEL_AVX2],
                (unsigned long long)dispatch.cycles[AI_KERNEL_FMA],
                caps.has_avx2, caps.has_fma, caps.avx_state_ready,
                measured_at);
        if (fclose(csv) != 0) {
            perror("error: unable to close csv");
            free(samples_raw);
            free(model);
            return 1;
        }
        printf("\n  Appended %s\n", csv_path);
    }

    free(samples_raw);
    free(model);
    return 0;
}
