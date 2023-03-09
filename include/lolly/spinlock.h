#pragma once

#include <atomic>

namespace Lolly {

class Spinlock {
private:
  std::atomic_flag flag = ATOMIC_FLAG_INIT;

public:
  void lock();
  void unlock();
};

} // namespace Lolly
