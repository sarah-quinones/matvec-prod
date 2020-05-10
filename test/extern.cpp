#include "simd.hpp"
#include "bench.hpp"

#define INSTANTIATE(T, NRows, NCols)                                           \
  template void eigen_::prod(                                                  \
      Mat<T, NRows, NCols> const&, Vec<T, NCols> const&, Vec<T, NRows>&);      \
  template void blaze_::prod(                                                  \
      Mat<T, NRows, NCols> const&, Vec<T, NCols> const&, Vec<T, NRows>&);      \
  template void matvec::prod(                                                  \
      Mat<T, NRows, NCols> const&, Vec<T, NCols> const&, Vec<T, NRows>&);

#define INSTANTIATE_4(T, NRows, NCols)                                         \
  INSTANTIATE(T, NRows, NCols)                                                 \
  INSTANTIATE(T, NRows + 1, NCols)                                             \
  INSTANTIATE(T, NRows + 2, NCols)                                             \
  INSTANTIATE(T, NRows + 3, NCols)

#define INSTANTIATE_16(T, NRows, NCols)                                        \
  INSTANTIATE_4(T, NRows, NCols)                                               \
  INSTANTIATE_4(T, NRows + 4, NCols)                                           \
  INSTANTIATE_4(T, NRows + 8, NCols)                                           \
  INSTANTIATE_4(T, NRows + 12, NCols)

#define INSTANTIATE_64(T, NRows, NCols)                                        \
  INSTANTIATE_16(T, NRows, NCols)                                              \
  INSTANTIATE_16(T, NRows + 16, NCols)                                         \
  INSTANTIATE_16(T, NRows + 32, NCols)                                         \
  INSTANTIATE_16(T, NRows + 48, NCols)

INSTANTIATE_64(FLOAT_TYPE, 0, NCOLS)
INSTANTIATE_64(FLOAT_TYPE, 64, NCOLS)
INSTANTIATE(FLOAT_TYPE, 128, NCOLS)
