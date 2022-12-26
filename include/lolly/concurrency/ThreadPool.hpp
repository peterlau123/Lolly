#include <future>
#include <thread>
#include <utility>

namespace Lolly {

class ThreadPool {
public:
  bool Init(int num_threads);

  ~ThreadPool();

  template <typename Func, typename... Args,
            typename R = typename std::result_of<Func(Args...)>::type>
  std::future<R> AddTask(Func &&f, Args &&... args);

  ThreadPool() = delete;

  ThreadPool(const ThreadPool &other_threadpool) = delete;

  ThreadPool &operator=(const ThreadPool &other_threadpool) = delete;

private:
  void Start(int n);

  void Stop();

  void ThreadHandle();

  struct Context;
  using ContextPtr = std::unique_ptr<Context>;
  ContextPtr impl_ptr_;
};
} // namespace Lolly
