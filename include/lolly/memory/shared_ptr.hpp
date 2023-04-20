#pragma once

namespace Lolly {

template <typename T, typename Allocator<T>> class shared_ptr {
public:
  shared_ptr() : {}

private:
};

template <typename T, typename... Args>
shared_ptr<T> make_shared(Args &&...args) {}

} // namespace Lolly
