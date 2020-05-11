#ifndef INCLUDE_UTILITY
#define INCLUDE_UTILITY
#include <climits>
#include <cstddef>
#include <utility>

#ifdef _MSC_VER
#define INLINE __forceinline inline
#define NOINLINE __declspec(noinline)
#define RESTRICT __restrict
#else
#define INLINE [[gnu::always_inline]] inline
#define NOINLINE [[gnu::noinline]] [[gnu::aligned(256)]]
#define RESTRICT __restrict__
#endif

using f32 = float;
using f64 = double;

static_assert(CHAR_BIT == 8);
static_assert(sizeof(f32) == 4);
static_assert(sizeof(f64) == 8);

template <int N> using int_constant = std::integral_constant<int, N>;

template <typename Fn, std::size_t Offset, std::size_t... Ns>
void for_each_impl(Fn const& fn, std::index_sequence<Ns...>) {
  (fn(std::integral_constant<int, Offset + Ns>{}), ...);
}
template <typename Fn, std::size_t... Ns>
INLINE void unroll_impl(Fn const& fn, int start, std::index_sequence<Ns...>) {
  (fn(start + static_cast<int>(Ns)), ...);
}

template <int Start, int End, typename Fn> void for_each(Fn const& fn) {
  for_each_impl<Fn, Start>(fn, std::make_index_sequence<End - Start>());
}

template <typename... Ts> void unused(Ts const&...){};

template <typename T>
using naked_type = std::remove_cv_t<std::remove_reference_t<T> >;

template <typename U, typename V>
using const_like = std::conditional_t<            //
    std::is_const_v<std::remove_reference_t<V> >, //
    naked_type<U> const,                          //
    naked_type<U>                                 //
    >;

template <int N_Per_Iter, int Start, int Count, typename Fn>
INLINE void unroll(Fn const& fn) {
  for (int i = Start; i < Start + Count / N_Per_Iter; ++i) {
    unroll_impl(fn, i * N_Per_Iter, std::make_index_sequence<N_Per_Iter>());
  }
  constexpr int rem = Count % N_Per_Iter;
  if constexpr (rem != 0) {
    unroll<rem, Start + Count - rem, rem>(fn);
  }
}
#endif
