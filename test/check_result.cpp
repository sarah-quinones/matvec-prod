#include "fmt/color.h"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "bench.hpp"
#include "simd.hpp"

#define EXTERN_4(T, NRows, NCols)                                              \
  EXTERN_TPL(T, NRows, NCols)                                                  \
  EXTERN_TPL(T, NRows + 1, NCols)                                              \
  EXTERN_TPL(T, NRows + 2, NCols)                                              \
  EXTERN_TPL(T, NRows + 3, NCols)

#define EXTERN_16(T, NRows, NCols)                                             \
  EXTERN_4(T, NRows, NCols)                                                    \
  EXTERN_4(T, NRows + 4, NCols)                                                \
  EXTERN_4(T, NRows + 8, NCols)                                                \
  EXTERN_4(T, NRows + 12, NCols)

#define EXTERN_64(T, NRows, NCols)                                             \
  EXTERN_16(T, NRows, NCols)                                                   \
  EXTERN_16(T, NRows + 16, NCols)                                              \
  EXTERN_16(T, NRows + 32, NCols)                                              \
  EXTERN_16(T, NRows + 48, NCols)

#define EXTERN_128(T, NRows, NCols)                                            \
  EXTERN_64(T, NRows, NCols)                                                   \
  EXTERN_64(T, NRows + 64, NCols)

EXTERN_128(f32, 0, 2)
EXTERN_128(f32, 0, 4)
EXTERN_128(f32, 0, 8)

EXTERN_128(f64, 0, 2)
EXTERN_128(f64, 0, 4)
EXTERN_128(f64, 0, 8)

template <typename T, int n_rows, int n_cols> void check_result() {
  Mat<T, n_rows, n_cols> mat{};
  Vec<T, n_cols> in{};
  Vec<T, n_rows> out{};
  Vec<T, n_rows> out_eigen{};
  Vec<T, n_rows> out_blaze{};

  blaze::setSeed(0);
  for (std::size_t j = 0; j < n_cols; ++j) {
    for (std::size_t i = 0; i < n_rows; ++i) {
      mat(i, j) = blaze::rand<T>();
    }
    in[j] = blaze::rand<T>();
  }

  eigen_::prod(mat, in, out_blaze);
  blaze_::prod(mat, in, out_eigen);
  matvec::prod(mat, in, out);

  T err1 = blaze::max(blaze::abs(out - out_eigen));
  T err2 = blaze::max(blaze::abs(out - out_blaze));
  T eps = 4 * std::numeric_limits<T>::epsilon();
  bool err_cond = err1 > eps or err2 > eps;

  auto fail = [] {
    fmt::print(
        fmt::bg(fmt::color::red) | fmt::fg(fmt::color::white), "{}", "fail");
    fmt::print("\n");
  };
  auto pass = [] {
    fmt::print(
        fmt::bg(fmt::color::cornflower_blue) | fmt::fg(fmt::color::white),
        "{}",
        "pass");
    fmt::print("\n");
  };

  fmt::print(
      "Testing [f{}][ {:>3}×{:>2} ] matrix : ",
      sizeof(T) * CHAR_BIT,
      n_rows,
      n_cols);
  if (err_cond) {
    fail();
    fmt::print(
        "matvec vs eigen: {}\n"
        "matvec vs blaze: {}\n",
        err1,
        err2);
    fmt::print(
        "matvec : {}\n"
        "blaze  : {}\n",
        blaze::trans(out),
        blaze::trans(out_blaze));
    std::exit(1);
  } else {
    pass();
  }
}

int main() {
  for_each<0, 128>([](auto i) { check_result<f32, decltype(i)::value, 2>(); });
  for_each<0, 128>([](auto i) { check_result<f32, decltype(i)::value, 4>(); });
  for_each<0, 128>([](auto i) { check_result<f32, decltype(i)::value, 8>(); });
  for_each<0, 128>([](auto i) { check_result<f64, decltype(i)::value, 2>(); });
  for_each<0, 128>([](auto i) { check_result<f64, decltype(i)::value, 4>(); });
  for_each<0, 128>([](auto i) { check_result<f64, decltype(i)::value, 8>(); });
}
