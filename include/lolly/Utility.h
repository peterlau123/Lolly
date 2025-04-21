#pragma once



#define _IN_
#define _OUT_
#define _INOUT_


namespace Lolly {

template <bool cond, typename T, typename U> class IfThenElse {};

template <true, typename T, typename U> class IfThenElse {
public:
  using Type = typename T;
};

template <false, typename T, typename U> class IfThenElse {
public:
  using Type = typename U;
};

} // namespace Lolly
