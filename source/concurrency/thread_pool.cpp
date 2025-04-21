#include "lolly/concurrency/thread_pool.h"

#include <condition_variable>
#include <cstdint>

#include <queue>

namespace Lolly {

struct ThreadPool::Context {
  int num_threads;

  std::vector<std::thread> threads;

  std::condition_variable cv;

  std::mutex task_que_mutex;

  struct Task {
    using Func = std::function<void()>;
    Task(Func &&in_f) : f(std::forward<Func>(in_f)) {}
    void operator()() { f(); };
    Func f;
  };
  std::queue<Task> task_queue;
};

bool ThreadPool::Init(int num_threads) {
  impl_ptr_ = std::unique_ptr<Context>(new Context);
  if (!impl_ptr_)
    return false;

  int hardware_thread_cnt = std::thread::hardware_concurrency();
  impl_ptr_->num_threads = (0 == hardware_thread_cnt) ? 4 : hardware_thread_cnt;

  Start(impl_ptr_->num_threads);
  return true;
}

template <typename Func, typename... Args, typename R>
std::future<R> ThreadPool::AddTask(Func &&f, Args &&...args) {
  std::packaged_task<R(Args...)> p_task(std::forward<Func>(f),
                                        std::forward<Args>(args)...);

  auto result = p_task.get_future();

  auto t = [&p_task]() { p_task(); };
  Context::Task task(t);
  impl_ptr_->task_queue.push(task);
  impl_ptr_->cv.notify_all();

  return result;
}

void ThreadPool::Start(int n) {
  for (int i = 0; i < n; i++) {
    impl_ptr_->threads.push_back(std::thread(&ThreadPool::ThreadHandle, this));
  }
}

void ThreadPool::ThreadHandle() {
  std::unique_lock<std::mutex> lk(impl_ptr_->task_que_mutex);
  while (true) {
    impl_ptr_->cv.wait(lk, [&]() -> bool {
      if (!impl_ptr_->task_queue.empty()) {
        return true;
      }
      return false;
    });
    auto t = impl_ptr_->task_queue.front();
    t();
    impl_ptr_->task_queue.pop();
    lk.unlock();
  }
}

void ThreadPool::Stop() {
  for (auto &&t : impl_ptr_->threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

ThreadPool::~ThreadPool() { Stop(); }

} // namespace Lolly
