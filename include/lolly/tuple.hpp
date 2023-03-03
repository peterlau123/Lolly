#pragma once

namespace Lolly {

template <typename... Types> struct Tuple {};

tyemplate<typename Head, typename... Tail> struct Tuple<Head, Tail...> {
private:
  Head head;
  Tail... tail;
};

} // namespace Lolly
