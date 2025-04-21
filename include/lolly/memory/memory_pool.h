#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>

namespace Lolly {

/*Fix-sized and thread-safe memory pool
 * */
class SimpleMemoryPool {
public:
  SimpleMemoryPool(size_t byte_count, int granuality);

  void *Get(size_t request_sz);

  void Put(void *data);

  ~SimpleMemoryPool();

  SimpleMemoryPool() = delete;

  SimpleMemoryPool(const SimpleMemoryPool &src_pool) = delete;

  SimpleMemoryPool &operator=(const SimpleMemoryPool &src_pool) = delete;

  SimpleMemoryPool(SimpleMemoryPool &&src_pool) = delete;

private:
  int _decideGranuality(int request_gran);

  struct Block {
    Block() : data(nullptr), byte_sz(0), free(true) {}
    Block(void *src_data, size_t cnt)
        : data(src_data), byte_sz(cnt), free(true) {}
    Block(const Block &other_block) {
      data = other_block.data;
      byte_sz = other_block.byte_sz;
      free = other_block.free;
    }
    void *data;
    size_t byte_sz;
    bool free;
  };

  int granuality_;
  int byte_count_;

  std::mutex mutex_;
  std::vector<Block> blocks_;
  std::unordered_map<int, int> used_buffer_;
  int ref_cnt_{0};
  struct Deleter {
    void operator()(uint8_t *p) {
      if (nullptr != p) {
        free(p);
      }
    }
  };
  std::unique_ptr<uint8_t, Deleter> pool_data_{nullptr, Deleter()};
};

/*
 *Memory pool support element index access
 * */
template <typename T> class IndexedMemoryPool {};

/*
 *No need to specify the total byte and granuality of the memory pool
 *It needs a first-run,after first-run,it collects the memory requirement of the
 *program and gives the best total byte and granuality for the program
 *
 *It has two advantages:
 *1.save manual tuning and give best choice for the program
 *2.adpative to many different programs
 * */
class AdaptiveMemoryPool {};

} // namespace Lolly
