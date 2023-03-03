#pragma once

namespace Lolly {

template <typename... Types> class TypeList {};

template <typename Head, typename... Types> class Front {
  using Type = Head;
};

} // namespace Lolly
