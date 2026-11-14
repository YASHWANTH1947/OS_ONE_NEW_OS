/*
 * AVX2 matvec kernel for the probe, compiled with -mavx2 (no FMA).
 *
 * Freestanding: no immintrin.h, only GCC vector extensions. Each translation
 * unit carries its own ISA flags; this is the same per-TU structure the
 * shipped dispatch engine uses.
 *
 * Alignment contract: callers guarantee 32-byte-aligned row bases. Both probe
 * shapes satisfy it (784*4 and 128*4 byte row strides are multiples of 32).
 */
#include <stdint.h>

typedef float v8 __attribute__((vector_size(32), may_alias));

static inline v8 vload(const float *p)
{
    return *(const v8 *)p;
}

/* Horizontal sum of 8 lanes using same-width shuffles only, so the code is
 * valid on any GCC that supports __builtin_shufflevector.
 *
 *   v   = (a0 a1 a2 a3 a4 a5 a6 a7)
 *   h   = (a4 a5 a6 a7 ...)        -> f1[0..3] = a0+a4 .. a3+a7
 *   g   = (f1[2] f1[3] ...)        -> f2[0..1] = sums of four halves
 *   k   = (f2[1] ...)              -> f3[0]    = full sum
 */
static inline float vsum8(v8 v)
{
    v8 h = __builtin_shufflevector(v, v, 4, 5, 6, 7, 0, 1, 2, 3);
    v8 f1 = v + h;
    v8 g = __builtin_shufflevector(f1, f1, 2, 3, 0, 1, 4, 5, 6, 7);
    v8 f2 = f1 + g;
    v8 k = __builtin_shufflevector(f2, f2, 1, 0, 2, 3, 4, 5, 6, 7);
    v8 f3 = f2 + k;
    return f3[0];
}

void matvec_avx2(const float *w, const float *x, const float *b, float *out,
                 uint32_t rows, uint32_t cols)
{
    uint32_t r;
    uint32_t c;

    for (r = 0; r < rows; ++r) {
        const float *row = &w[r * cols];
        /* The bias rides in lane 0 and survives the horizontal sum. */
        v8 acc = {b[r], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float tail = 0.0f;

        for (c = 0; c + 7u < cols; c += 8u) {
            acc += vload(x + c) * vload(row + c);
        }
        for (; c < cols; ++c) {
            tail += row[c] * x[c];
        }
        out[r] = vsum8(acc) + tail;
    }
}
