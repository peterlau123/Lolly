#include "lolly/concurrency/ThreadPool.hpp"

namespace Lolly {

struct ThreadPool::Context {
  int num_threads;

  std::vector<std::thread> threads;

  std::condition_variable cv;

  std::mutex task_que_mutex;

  struct Task {
    using Func = std::function<void()>;
    Task(Func &&in_f) : f(std::forward<Func>(in_f)) {}
    Func f;
  };
  std::queue<Task> task_queue;
};

ThreadPool::ThreadPool() : impl_ptr_(nullptr) {}

bool ThreadPool::Init(int num_threads) {
  impl_ptr_ = std::make_shared<Context>();
  if (!impl_ptr_)
    returnn false;

  int hardware_thread_cnt = std::thread::hardware_concurrency();
  impl_ptr_->num_threads = (0 == hardware_thread_cnt) ? 4 : hardware_thread_cnt;

  Start(impl_ptr_->num_threads);
}

template <typename Func, typename... Args,
          typename R = std::result_of<Func(Args...)>::type>
std::future<R> ThreadPool::AddTask(Func &&f, Args... &&args) {
  std::packaged_task<R(Args...)> p_task(std::forward<Func &&>(f),
                                        std::forward<Args... &&>(args));

  auto result = p_task.get_future();

  auto t = [&p_task]() { p_task(); };
  Context::Task task(t);
  impl_ptr->task_queue_.push(task);
  impl_ptr_->cv.notify_all();

  return result;
}

ThreadPool::Start(int n) {
  threads_.push_back(std::thread(&ThreadPool::ThreadHandle, this));
}

void ThreadPool::ThreadHandle() {
  // TODO
}

void ThreadPool::Stop() {
  for (auto &&t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

ThreadPool::~ThreadPool() { Stop(); }
} // namespace Lolly
