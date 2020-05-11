#include "bench.hpp"

EXTERN_TPL(FLOAT_TYPE, NCOLS, NROWS);
RUN_BENCHMARKS(FLOAT_TYPE, NCOLS, NROWS);

#ifdef BM_EIGEN
#define METHOD eigen
#endif

#ifdef BM_BLAZE
#define METHOD blaze
#endif

#ifdef BM_SIMD
#define METHOD simd
#endif

#define CAT0(x, y) x##_##y
#define CAT(x, y) CAT0(x, y)
#define STRINGIZE0(x) #x
#define STRINGIZE(x) STRINGIZE0(x)

int main() {
  run_bench(STRINGIZE(CAT(CAT(CAT(FLOAT_TYPE, NCOLS), NROWS), METHOD)));
}
