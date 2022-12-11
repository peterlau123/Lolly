#include "lolly/concurrency/ThreadPool.hpp"

namespace Lolly {

  struct ThreadPool::TaskBase {
  public:
    virtual void Run() = 0;
  };

  template <typename TaskT> struct ThreadPool::Task : public ThreadPool::TaskBase {
  public:
    Task(TaskT&& t) { t_ = std::forward<TaskT&&>(t); }

    void Run() override { t_(); }

  private:
    TaskT&& t_:
  };

  ThreadPool::ThreadPool() : num_threads_(4) {
    int hardware_thread_cnt = std::thread::hardware_concurrency();
    num_threads_ = (0 == hardware_thread_cnt) ? num_threads_ : hardware_thread_cnt;

    Start(num_threads_);
  }

  template <typename Func, typename... Args, typename R = std::result_of<Func(Args...)>::type>
  std::future<R> AddTask(Func&& f, Args...&& args) {
    std::packaged_task<R(Args...)> p_task(std : forward<Func&&>(f), std::forward<Args...&&>(args));

    ThreadPool::Task task(p_task);
    task_queue_.push(task);
    cv_.notify_all();

    auto result = p_task.get_future();
    return result;
  }

  ThreadPool::Start(int n) { threads_.push_back(std::thread(&ThreadPool::ThreadHandle, this)); }

  void ThreadPool::ThreadHandle() {
    cv_.wait();
    while (!task_queue_.empty()) {
      auto task = task_queue_.front();
      task->Run();
      task_que_mutex_.pop();
    }
  }

  void ThreadPool::Stop() {
    for (auto&& t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  ThreadPool::~ThreadPool() { Stop(); }
}  // namespace Lolly
