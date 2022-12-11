#include <condition_variable>
#include <thread>
#include <utility>

namespace Lolly {

  class SimpleThreadPool {
  public:
    SimpleThreadPool();

    ~SimpleThreadPool();

    template <typename Func, typename... Args> AddTask(Func f, Args... args);

    SimpleThreadPool() = delete;

    SimpleThreadPool(const SimpleThreadPool& other_threadpool) = delete;

    SimpleThreadPool& operator=(const SimpleThreadPool& other_threadpool) = delete;

  private:
    void Start(int n);

    void Stop();

    void ThreadHandle();

    int num_threads_;
    std::vector<std::thread> threads_;

    std::condition_variable cv_;

    std::mutex task_que_mutex_;

    struct TaskBase;
    using TaskHandle = std::unique_ptr<TaskBase>;
    std::queue<TaskHandle> task_queue_;
  };
}  // namespace Lolly
