#ifndef INCLUDE_SIMD
#define INCLUDE_SIMD
#include <x86intrin.h>
#include "utility.hpp"

// Let the compiler unroll the loops for now
constexpr int UnrollNum = 1;

// [f32][n_rows][8]
template <int n_rows>
NOINLINE void matvec_simd(
    f32 const* mat, f32 const* in_data, f32* out_data, int_constant<8>) {
  static_assert(sizeof(f32) * 8 % 32 == 0);
  __m256 in;
  __m256 mat_row_1;
  __m256 mat_row_2;
  __m256 mat_row_3;
  __m256 mat_row_4;
  __m128 f4;
  __m128 f4_2;
  in = _mm256_loadu_ps(in_data);

  // 4 rows at a time, at row 4xi
  auto batch_4 = [&](int i) {
    mat_row_1 = _mm256_loadu_ps(mat + 32 * i);
    mat_row_2 = _mm256_loadu_ps(mat + 32 * i + 8);
    mat_row_3 = _mm256_loadu_ps(mat + 32 * i + 16);
    mat_row_4 = _mm256_loadu_ps(mat + 32 * i + 24);

    mat_row_1 = mat_row_1 * in; // a0 a1 a2 a3 a4 a5 a6 a7
    mat_row_2 = mat_row_2 * in; // b0 b1 b2 b3 b4 b5 b6 b7
    mat_row_3 = mat_row_3 * in; // c0 c1 c2 c3 c4 c5 c6 c7
    mat_row_4 = mat_row_4 * in; // d0 d1 d2 d3 d4 d5 d6 d7

    // a0 + a1 | a2 + a3 | b0 + b1 | b2 + b3 | =>
    // a4 + a5 | a6 + a7 | b4 + b5 | b6 + b7
    mat_row_1 = _mm256_hadd_ps(mat_row_1, mat_row_2);

    // c0 + c1 | c2 + c3 | d0 + d1 | d2 + d3 | =>
    // c4 + c5 | c6 + c7 | d4 + d5 | d6 + d7
    mat_row_3 = _mm256_hadd_ps(mat_row_3, mat_row_4);

    // a0..4 | b0..4 | c0..4 | d0..4 | =>
    // a4..8 | b4..8 | c4..8 | d4..8
    mat_row_1 = _mm256_hadd_ps(mat_row_1, mat_row_3);

    // a4..8 | b4..8 | c4..8 | d4..8
    f4 = _mm256_extractf128_ps(mat_row_1, 1);
    // a0..8 | b0..8 | c0..8 | d0..8
    f4 = f4 + _mm256_castps256_ps128(mat_row_1);

    _mm_storeu_ps(out_data + 4 * i, f4);
  };

  auto batch_2 = [&](int i) {
    mat_row_1 = _mm256_loadu_ps(mat + 16 * i);
    mat_row_2 = _mm256_loadu_ps(mat + 16 * i + 8);

    mat_row_1 = mat_row_1 * in; // a0 a1 a2 a3 a4 a5 a6 a7
    mat_row_2 = mat_row_2 * in; // b0 b1 b2 b3 b4 b5 b6 b7

    // a0 + a1 | a2 + a3 | b0 + b1 | b2 + b3 | =>
    // a4 + a5 | a6 + a7 | b4 + b5 | b6 + b7
    mat_row_1 = _mm256_hadd_ps(mat_row_1, mat_row_2);

    // a4 + a5 | a6 + a7 | b4 + b5 | b6 + b7
    f4 = _mm256_extractf128_ps(mat_row_1, 1);

    // a0 + a1 + a4 + a5 | a2 + a3 + a6 + a7 | =>
    // b0 + b1 + b4 + b5 | b2 + b3 + b6 + b7
    f4 = f4 + _mm256_castps256_ps128(mat_row_1);

    f4 = _mm_hadd_ps(f4, f4);

    _MM_EXTRACT_FLOAT(out_data[2 * i], f4, 0);
    _MM_EXTRACT_FLOAT(out_data[2 * i + 1], f4, 1);
  };

  auto batch_1 = [&](int i) {
    mat_row_1 = _mm256_loadu_ps(mat + 8 * i);

    mat_row_1 = mat_row_1 * in; // a0 a1 a2 a3 a4 a5 a6 a7

    // a0 + a4 | a1 + a5 | a2 + a6 | a3 + a7
    f4 =
        _mm256_extractf128_ps(mat_row_1, 1) + _mm256_castps256_ps128(mat_row_1);

    //  a2 + a6 | a3 + a7 | a0 + a4 | a1 + a5
    f4_2 = _mm_permute_ps(f4, 0x4E);

    //  a0 + a2 + a4 + a6 | a1 + a3 + a5 + a7 | =>
    //  a0 + a2 + a4 + a6 | a1 + a3 + a5 + a7
    f4 = f4 + f4_2;

    // a1 + a3 + a5 + a7 | a1 + a3 + a5 + a7 | =>
    // a1 + a3 + a5 + a7 | a1 + a3 + a5 + a7
    f4_2 = _mm_movehdup_ps(f4);

    f4 = f4 + f4_2;
    _MM_EXTRACT_FLOAT(out_data[i], f4, 0);
  };
  unroll<UnrollNum, 0, n_rows / 4>(batch_4);

  if constexpr (n_rows % 4 >= 2) {
    batch_2(n_rows / 2 - 1);
  }
  if constexpr (n_rows % 2 == 1) {
    batch_1(n_rows - 1);
  }
}

// [f32][n_rows][4]
template <int n_rows>
NOINLINE void matvec_simd(
    f32 const* mat, f32 const* in_data, f32* out_data, int_constant<4>) {
  __m256 in;
  __m256 mat_row_1;
  __m256 mat_row_2;
  __m128 f4;
  __m128 f4_2;
  in = _mm256_loadu2_m128(in_data, in_data);

  // 4 rows at a time, at row 4xi
  auto batch_4 = [&](int i) {
    mat_row_1 = _mm256_loadu_ps(mat + 16 * i);
    mat_row_2 = _mm256_loadu_ps(mat + 16 * i + 8);

    // a0 a1 a2 a3 | b0 b1 b2 b3
    mat_row_1 = mat_row_1 * in;
    // c0 c1 c2 c3 | d0 d1 d2 d3
    mat_row_2 = mat_row_2 * in;

    // a0 + a1 | a2 + a3 | c0 + c1 | c2 + c3 | =>
    // b0 + b1 | b2 + b3 | d0 + d1 | d2 + d3
    mat_row_1 = _mm256_hadd_ps(mat_row_1, mat_row_2);
    f4 = __builtin_shufflevector(mat_row_1, mat_row_1, 0, 4, 2, 6);
    f4_2 = __builtin_shufflevector(mat_row_1, mat_row_1, 1, 5, 3, 7);
    f4 = f4 + f4_2;
    _mm_storeu_ps(out_data + 4 * i, f4);
  };

  auto batch_1 = [&](int i) {
    f4 = _mm_loadu_ps(mat + 4 * i);

    // a0 a1 a2 a3
    f4 = _mm_loadu_ps(in_data) * f4;
    // a0 a0 a2 a2
    f4_2 = _mm_movehdup_ps(f4);
    // a0 + a0 | a0 + a1 | a2 + a2 | a2 + a3
    f4_2 = f4 + f4_2;

    // a0 + a0 | a0 + a1 | a0 + a0 | a0 + a1
    f4 = _mm_movehl_ps(f4_2, f4_2);

    f4 = f4 + f4_2;
    _MM_EXTRACT_FLOAT(out_data[i], f4, 0);
  };
  unroll<UnrollNum, 0, n_rows / 4>(batch_4);

  if constexpr (n_rows % 4 >= 2) {
    batch_1((n_rows / 2) * 2 - 2);
    batch_1((n_rows / 2) * 2 - 1);
  }
  if constexpr (n_rows % 2 == 1) {
    batch_1(n_rows - 1);
  }
}

// [f32][n_rows][2]
template <int n_rows>
NOINLINE void matvec_simd(
    f32 const* mat, f32 const* in_data, f32* out_data, int_constant<2>) {
  __m256 in;
  __m256 mat_row_1;
  __m256 mat_row_2;
  __m128 f4;
  __m128 f4_2;

  in[0] = in_data[0];
  in[1] = in_data[1];
  in[2] = in_data[0];
  in[3] = in_data[1];
  in[4] = in_data[0];
  in[5] = in_data[1];
  in[6] = in_data[0];
  in[7] = in_data[1];

  // 8 rows at a time, at row 8xi
  auto batch_8 = [&](int i) {
    mat_row_1 = _mm256_loadu_ps(mat + 16 * i);
    mat_row_2 = _mm256_loadu_ps(mat + 16 * i + 8);

    // a0 a1 b0 b1 c0 c1 d0 d1
    mat_row_1 = mat_row_1 * in;
    // e0 e1 f0 f1 g0 g1 h0 h1
    mat_row_2 = mat_row_2 * in;
    // a b e f c d g h
    mat_row_1 = _mm256_hadd_ps(mat_row_1, mat_row_2);
    mat_row_1 =
        __builtin_shufflevector(mat_row_1, mat_row_1, 0, 1, 4, 5, 2, 3, 6, 7);
    _mm256_storeu_ps(out_data + 8 * i, mat_row_1);
  };

  // 4 rows at a time, at row 4xi
  auto batch_4 = [&](int i) {
    mat_row_1 = _mm256_loadu_ps(mat + 8 * i);

    // a0 a1 b0 b1 c0 c1 d0 d1
    mat_row_1 = mat_row_1 * in;
    f4 = __builtin_shufflevector(mat_row_1, mat_row_1, 0, 2, 4, 6);
    f4_2 = __builtin_shufflevector(mat_row_1, mat_row_1, 1, 3, 5, 7);
    f4 = f4 + f4_2;
    _mm_storeu_ps(out_data + 4 * i, f4);
  };

  auto batch_2 = [&](int i) {
    f4 = _mm_loadu_ps(mat + 4 * i);
    f4_2[0] = in_data[0];
    f4_2[1] = in_data[1];
    f4_2[2] = in_data[0];
    f4_2[3] = in_data[1];
    // a1 a2 b1 b2
    f4 = f4 * f4_2;
    out_data[2 * i] = f4[0] + f4[1];
    out_data[2 * i + 1] = f4[2] + f4[3];
  };
  unused(batch_2);

  auto batch_1 = [&](int i) {
    out_data[i] = in_data[0] * mat[2 * i] + in_data[1] * mat[2 * i + 1];
  };
  unroll<UnrollNum, 0, n_rows / 8>(batch_8);

  if constexpr (n_rows % 8 >= 4) {
    batch_4(n_rows / 4 - 1);
  }

  if constexpr (n_rows % 4 >= 2) {
    // batch_2(n_rows / 2 - 1);
    batch_1((n_rows / 2) * 2 - 2);
    batch_1((n_rows / 2) * 2 - 1);
  }
  if constexpr (n_rows % 2 == 1) {
    batch_1(n_rows - 1);
  }
}

// [f64][n_rows][8]
template <int n_rows>
NOINLINE void matvec_simd(
    f64 const* mat, f64 const* in_data, f64* out_data, int_constant<8>) {
  static_assert(sizeof(f64) * 4 % 32 == 0);
  __m256d in_1;
  __m256d in_2;
  __m256d mat_row_11;
  __m256d mat_row_12;
  __m256d mat_row_21;
  __m256d mat_row_22;
  __m128d sums;
  in_1 = _mm256_loadu_pd(in_data);
  in_2 = _mm256_loadu_pd(in_data + 4);

  auto batch_2 = [&](int i) {
    mat_row_11 = _mm256_loadu_pd(mat + 16 * i);
    mat_row_12 = _mm256_loadu_pd(mat + 16 * i + 4);
    mat_row_21 = _mm256_loadu_pd(mat + 16 * i + 8);
    mat_row_22 = _mm256_loadu_pd(mat + 16 * i + 12);

    mat_row_11 = in_1 * mat_row_11;
    mat_row_11 = _mm256_fmadd_pd(in_2, mat_row_12, mat_row_11);

    mat_row_21 = in_1 * mat_row_21;
    mat_row_21 = _mm256_fmadd_pd(in_2, mat_row_22, mat_row_21);

    // a1 + a2 + a5 + a6 | b... | a3 + a4 + a7 + a8 | b...
    mat_row_11 = _mm256_hadd_pd(mat_row_11, mat_row_21);
    sums = _mm256_extractf128_pd(mat_row_11, 1);
    sums = sums + _mm256_castpd256_pd128(mat_row_11);
    _mm_storeu_pd(out_data + 2 * i, sums);
  };

  auto batch_1 = [&](int i) {
    mat_row_11 = _mm256_loadu_pd(mat + 8 * i);
    mat_row_12 = _mm256_loadu_pd(mat + 8 * i + 4);

    // a1 a2 a3 a4
    mat_row_11 = in_1 * mat_row_11;
    // a5 a6 a7 a8
    mat_row_12 = in_2 * mat_row_12;

    // a1 + a5 | a2 + a6 | a3 + a7 | a4 + a8
    mat_row_11 = mat_row_11 + mat_row_12;

    sums = _mm256_extractf128_pd(mat_row_11, 1);

    // 1 + 2 + 5 + 6 | 3 + 4 + 7 + 8
    sums = sums + _mm256_castpd256_pd128(mat_row_11);
    out_data[i] = sums[0] + sums[1];
  };

  unroll<UnrollNum, 0, n_rows / 2>(batch_2);

  if constexpr (n_rows % 2 == 1) {
    batch_1(n_rows - 1);
  }
}

// [f64][n_rows][4]
template <int n_rows>
NOINLINE void matvec_simd(
    f64 const* mat, f64 const* in_data, f64* out_data, int_constant<4>) {
  static_assert(sizeof(f64) * 4 % 32 == 0);
  __m256d in;
  __m256d mat_row_1;
  __m256d mat_row_2;
  __m128d sums;

  in = _mm256_loadu_pd(in_data);
  // 2 rows at a time, starting at row 2*i
  auto batch_2 = [&](int i) {
    mat_row_1 = _mm256_loadu_pd(mat + 8 * i);
    mat_row_2 = _mm256_loadu_pd(mat + 8 * i + 4);

    mat_row_1 = in * mat_row_1;
    mat_row_2 = in * mat_row_2;

    mat_row_1 = _mm256_hadd_pd(mat_row_1, mat_row_2);
    sums = _mm256_extractf128_pd(mat_row_1, 1);
    sums = sums + _mm256_castpd256_pd128(mat_row_1);
    _mm_storeu_pd(out_data + 2 * i, sums);
  };
  auto batch_1 = [&](int i) {
    mat_row_1 = _mm256_loadu_pd(mat + 4 * (i));
    mat_row_1 = in * mat_row_1;
    sums =
        _mm256_extractf128_pd(mat_row_1, 1) + _mm256_castpd256_pd128(mat_row_1);
    out_data[i] = sums[0] + sums[1];
  };

  unroll<UnrollNum, 0, n_rows / 2>(batch_2);

  if constexpr (n_rows % 2 == 1) {
    batch_1(n_rows - 1);
  }
}

// [f64][n_rows][2]
template <int n_rows>
NOINLINE void matvec_simd(
    f64 const* mat, f64 const* in_data, f64* out_data, int_constant<2>) {
  static_assert(sizeof(f64) * 4 % 32 == 0);
  __m256d in;
  __m256d mat_row_12;
  __m256d mat_row_34;
  __m128d in_2;
  __m128d d2_1;
  __m128d d2_2;
  in = _mm256_loadu2_m128d(in_data, in_data);

  // 4 rows at a time, starting at row 4*i
  auto batch_4 = [&](int i) {
    mat_row_12 = _mm256_loadu_pd(mat + 8 * i);
    mat_row_34 = _mm256_loadu_pd(mat + 8 * i + 4);

    // a0 a1 b0 b1
    mat_row_12 = in * mat_row_12;
    // c0 c1 d0 d1
    mat_row_34 = in * mat_row_34;

    // a0 + a1 | c0 + c1 | b0 + b1 | d0 + d1
    mat_row_12 = _mm256_hadd_pd(mat_row_12, mat_row_34);

    // a0 + a1 | b0 + b1 | c0 + c1 | d0 + d1
    f64 d = mat_row_12[2];
    mat_row_12[2] = mat_row_12[1];
    mat_row_12[1] = d;

    _mm256_storeu_pd(out_data + 4 * i, mat_row_12);
  };

  // 2 rows at a time, starting at row 2*i
  auto batch_2 = [&](int i) {
    d2_1 = _mm_loadu_pd(mat + 4 * i);
    d2_1 = d2_1 * in_2;
    d2_2 = _mm_permute_pd(d2_1, 1);
    // auto* write_ptr = static_cast<f64*>(
    //     __builtin_assume_aligned(out_data + 2 * i, 2 * sizeof(f64)));
    auto* write_ptr = out_data + 2 * i;
    write_ptr[0] = (d2_1 + d2_2)[0];

    d2_1 = _mm_loadu_pd(mat + 4 * i + 2);
    d2_1 = d2_1 * in_2;
    d2_2 = _mm_permute_pd(d2_1, 1);
    write_ptr[1] = (d2_1 + d2_2)[0];
  };
  // 2 rows at a time, starting at row 2*i
  auto batch_1 = [&](int i) {
    d2_1 = _mm_loadu_pd(mat + 2 * i);
    d2_1 = d2_1 * _mm_loadu_pd(in_data);
    out_data[i] = d2_1[0] + d2_1[1];
  };

  if constexpr (n_rows % 4 == 0) {
    unroll<UnrollNum, 0, n_rows / 4>(batch_4);
  } else if constexpr (n_rows % 4 == 1 and n_rows < 12) {
    in_2 = _mm_loadu_pd(in_data);
    unroll<UnrollNum, 0, n_rows / 2>(batch_2);
  } else {
    unroll<UnrollNum, 0, n_rows / 4>(batch_4);
    if constexpr (n_rows % 4 >= 2) {
      in_2 = _mm_loadu_pd(in_data);
      batch_2(n_rows / 2 - 1);
    }
  }
  if constexpr (n_rows % 2 == 1) {
    batch_1(n_rows - 1);
  }
}
#endif
