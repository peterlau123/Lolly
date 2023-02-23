#pragma once

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
template <typename... Types> class MyVariant {};

} // namespace Lolly
