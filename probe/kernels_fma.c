/*
 * AVX2 + FMA matvec kernel for the probe, compiled with -mavx2 -mfma.
 *
 * Same freestanding conventions as kernels_avx2.c: GCC vector extensions
 * rather than immintrin.h, and per-TU ISA flags. The FMA path folds the bias
 * in through a genuine fused multiply-add chain: acc = row[c] * x[c] + acc.
 */
#include <stdint.h>

typedef float v8 __attribute__((vector_size(32), may_alias));

static inline v8 vload(const float *p)
{
    return *(const v8 *)p;
}

static inline v8 vfma(v8 a, v8 b, v8 c)
{
    return __builtin_ia32_vfmaddps256(a, b, c);
}

/* Horizontal sum, same portable fold as the AVX2 TU. */
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

void matvec_fma(const float *w, const float *x, const float *b, float *out,
                uint32_t rows, uint32_t cols)
{
    uint32_t r;
    uint32_t c;

    for (r = 0; r < rows; ++r) {
        const float *row = &w[r * cols];
        v8 acc = {b[r], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float tail = 0.0f;

        for (c = 0; c + 7u < cols; c += 8u) {
            acc = vfma(vload(row + c), vload(x + c), acc);
        }
        for (; c < cols; ++c) {
            tail += row[c] * x[c];
        }
        out[r] = vsum8(acc) + tail;
    }
}
