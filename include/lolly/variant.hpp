#pragma once

<<<<<<< HEAD
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

=======
namespace Lolly {

template <typename T> class VariantStorage {};

template <typename... Types> class VariantChoice {};

/*
 * class MyVariant
 *
 * Dsicriminated unions
 *
 * Usage:
 * MyVariant<int,double,int> v;
 * v=2.0f;
 * std::cout<<v<<std::endl;
 *
 * */
>>>>>>> origin/feat-add_cache_policy
template <typename... Types> class MyVariant {};

} // namespace Lolly
