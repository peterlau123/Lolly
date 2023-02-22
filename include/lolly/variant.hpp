#pragma once

#include "TypeList.hpp"
#include "Utility.hpp"

namespace Lolly {

template<typename ...Types>
class LargestTypeOf{
public:
  using Type = IfThenElse<, , >::Type;
}

template <typename... Types> class VariantStorage {
private:
  using LargestT = LargestTypeOf(Types...)::Type;

  aligns(Types...) unsigned char buffer[sizeof(LargestT)];

public:
  void *getRawBuffer() { return buffer; }

  template <typename T> const T *getBufferAs const {
    return static_cast<const T *>(buffer);
  }

  template <typename T> T *getBufferAs { return static_cast<T *>(buffer); }
};

template <typename... Types> class VariantChoice {};

template <typename... Types> class MyVariant {};

} // namespace Lolly
