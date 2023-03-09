#include "spinlock.h"

namespace Lolly {

Spinlock::lock() {
  while (flag.test_and_set(std::memory_order_acquire)) { // acquire lock
    // Since C++20, it is possible to update atomic_flag's
    // value only when there is a chance to acquire the lock.
    // See also: https://stackoverflow.com/questions/62318642
#if defined(__cpp_lib_atomic_flag_test)
    while (flag.test(std::memory_order_relaxed)) // test lock
#endif
      ; // spin
  }
}

Spinlock::unlock() {

  flag.clear(std::memory_order_release); // release lock
}

} // namespace Lolly
