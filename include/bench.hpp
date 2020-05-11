#ifndef INCLUDE_BENCH
#define INCLUDE_BENCH

#include <fstream>
#include <iostream>

#include "benchmark/benchmark.h"

#include "blaze/math/StaticMatrix.h"
#include "blaze/math/StaticVector.h"

#include "Eigen/Core"

#include "simd.hpp"

#define SWALLOW_SEMICOLON struct unused_with_placeholder_id_##__LINE__

#ifdef BM_EIGEN
#define BENCH_EIGEN(...)                                                       \
  BENCHMARK_TEMPLATE(__VA_ARGS__);                                             \
  SWALLOW_SEMICOLON
#else
#define BENCH_EIGEN(...) SWALLOW_SEMICOLON
#endif

#ifdef BM_BLAZE
#define BENCH_BLAZE(...)                                                       \
  BENCHMARK_TEMPLATE(__VA_ARGS__);                                             \
  SWALLOW_SEMICOLON
#else
#define BENCH_BLAZE(...) SWALLOW_SEMICOLON
#endif

#ifdef BM_SIMD
#define BENCH_SIMD_(...)                                                       \
  BENCHMARK_TEMPLATE(__VA_ARGS__);                                             \
  SWALLOW_SEMICOLON
#else
#define BENCH_SIMD_(...) SWALLOW_SEMICOLON
#endif

template <
    typename T,                                                          //
    int n_rows,                                                          //
    int n_cols,                                                          //
    typename M = blaze::StaticMatrix<T, n_rows, n_cols, blaze::rowMajor> //
    >
using Mat = blaze::StaticMatrix< //
    T,                           //
    n_rows,                      //
    n_cols,                      //
    blaze::rowMajor,             //
    blaze::unaligned,            //
    blaze::unpadded              //
    >;

template <typename T, int size>
using Vec = blaze::StaticVector<T, size, blaze::columnVector, blaze::unaligned>;

template <typename T> auto row_maj_view(T&& mat) {
  using namespace Eigen;
  using U = naked_type<T>;
  using E = typename U::ElementType;
  return //
      Map<const_like<Matrix<E, U::rows(), U::columns(), RowMajor>, T>,
          Unaligned>{mat.data()};
}

template <typename T> auto vec_view(T&& mat) {
  using namespace Eigen;
  using U = naked_type<T>;
  using E = typename U::ElementType;
  return Map<const_like<Matrix<E, U::size(), 1>, T>, Unaligned>{mat.data()};
}

struct eigen_ {
  template <typename T, typename In, typename Out>
  NOINLINE static void //
  prod(T const& matrix, In const& in, Out& out);
};
struct blaze_ {
  template <typename M, typename In, typename Out>
  NOINLINE static void //
  prod(M const& matrix, In const& in, Out& out);
};
struct loop_ {
  template <typename T, typename In, typename Out>
  NOINLINE static void //
  prod(T const& matrix, In const& in, Out& out);
};
struct matvec {
  template <typename T, typename In, typename Out>
  NOINLINE static void //
  prod(T const& matrix, In const& in, Out& out);
};

template <typename T, typename In, typename Out>
void eigen_::prod(T const& matrix, In const& in, Out& out) {
  vec_view(out).noalias() = row_maj_view(matrix) * vec_view(in);
}

template <typename M, typename In, typename Out>
void blaze_::prod(M const& matrix, In const& in, Out& out) {
  out = blaze::noalias(matrix * in);
}

template <typename T, typename In, typename Out>
void loop_::prod(T const& matrix, In const& in, Out& out) {
  using E = typename T::ElementType;
  constexpr int Rows = T::rows();
  constexpr int Cols = T::columns();

  E const* RESTRICT mat_data = matrix.data();
  E const* RESTRICT in_data = in.data();
  E* RESTRICT out_data = out.data();

  for (int i = 0; i < Rows; ++i) {
    out_data[i] = 0;
    for (int j = 0; j < Cols; ++j) {
      out_data[i] += mat_data[i * Cols + j] * in_data[j];
    }
  }
}

template <typename T, typename In, typename Out>
void matvec::prod(T const& matrix, In const& in, Out& out) {
  matvec_simd<T::rows()>(          //
      matrix.data(),               //
      in.data(),                   //
      out.data(),                  //
      int_constant<T::columns()>{} //
  );
}

template <typename Method, typename T, int n_rows, int n_cols>
void bm(benchmark::State& state) {
  alignas(32) Mat<T, n_rows, n_cols> mat{};
  alignas(32) Vec<T, n_cols> in{};
  alignas(32) Vec<T, n_rows> out{};
  for (const auto& _ : state) {
    unused(_);
    benchmark::DoNotOptimize(mat);
    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
    Method::prod(mat, in, out);
    benchmark::ClobberMemory();
  }
}

template <typename T, int n_rows, int n_cols>
NOINLINE void bm_eigen(benchmark::State& state) {
  bm<eigen_, T, n_rows, n_cols>(state);
}
template <typename T, int n_rows, int n_cols>
NOINLINE void bm_blaze(benchmark::State& state) {
  bm<blaze_, T, n_rows, n_cols>(state);
}

template <typename T, int n_rows, int n_cols>
NOINLINE void bm_loop_(benchmark::State& state) {
  bm<loop_, T, n_rows, n_cols>(state);
}

template <typename T, int n_rows, int n_cols>
NOINLINE void bm_simd_(benchmark::State& state) {
  bm<matvec, T, n_rows, n_cols>(state);
}

void run_bench(const std::string& name);

#define EXTERN_TPL(T, NRows, NCols)                                            \
  extern template void eigen_::prod(                                           \
      Mat<T, NRows, NCols> const&, Vec<T, NCols> const&, Vec<T, NRows>&);      \
  extern template void blaze_::prod(                                           \
      Mat<T, NRows, NCols> const&, Vec<T, NCols> const&, Vec<T, NRows>&);      \
  extern template void matvec::prod(                                           \
      Mat<T, NRows, NCols> const&, Vec<T, NCols> const&, Vec<T, NRows>&)

#define EXTERN_4(T, NRows, NCols)                                              \
  EXTERN_TPL(T, (NRows), (NCols));                                             \
  EXTERN_TPL(T, (NRows) + 1, (NCols));                                         \
  EXTERN_TPL(T, (NRows) + 2, (NCols));                                         \
  EXTERN_TPL(T, (NRows) + 3, (NCols))

#define EXTERN_16(T, NRows, NCols)                                             \
  EXTERN_4(T, (NRows), (NCols));                                               \
  EXTERN_4(T, (NRows) + 4, (NCols));                                           \
  EXTERN_4(T, (NRows) + 8, (NCols));                                           \
  EXTERN_4(T, (NRows) + 12, (NCols))

#define EXTERN_64(T, NRows, NCols)                                             \
  EXTERN_16(T, (NRows), (NCols));                                              \
  EXTERN_16(T, (NRows) + 16, (NCols));                                         \
  EXTERN_16(T, (NRows) + 32, (NCols));                                         \
  EXTERN_16(T, (NRows) + 48, (NCols))

#define EXTERN_128(T, NRows, NCols)                                            \
  EXTERN_64(T, (NRows), (NCols));                                              \
  EXTERN_64(T, (NRows) + 64, (NCols))

#define EXTERN_128_ALL                                                         \
  EXTERN_128(f32, 0, 2);                                                       \
  EXTERN_128(f32, 0, 4);                                                       \
  EXTERN_128(f32, 0, 8);                                                       \
                                                                               \
  EXTERN_128(f64, 0, 2);                                                       \
  EXTERN_128(f64, 0, 4);                                                       \
  EXTERN_128(f64, 0, 8)

#define RUN_BENCHMARKS(T, NCols, NRows)                                        \
  BENCH_BLAZE(bm_blaze, T, (NRows), (NCols));                                  \
  BENCH_EIGEN(bm_eigen, T, (NRows), (NCols));                                  \
  BENCH_SIMD_(bm_simd_, T, (NRows), (NCols))

#endif
