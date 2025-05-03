#pragma once

#define _IN_
#define _OUT_
#define _INOUT_

namespace Lolly {

template <bool cond, typename T, typename U> struct IfThenElse {};

template <typename T, typename U> struct IfThenElse<true, T, U> {
public:
  using Type = T;
};

template <typename T, typename U> struct IfThenElse<false, T, U> {
public:
  using Type = U;
};

} // namespace Lolly
