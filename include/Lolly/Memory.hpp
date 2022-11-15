namespace Lolly {

/*Fix-sized and thread-safe memory pool
 * */
class SimpleMemoryPool {
 public:
  SimpleMemoryPool(size_t byte_count, int granuality);

  void *Get(size_t request_sz);

  void Put(void *data);

  ~SimpleMemoryPool();

  SimpleMemoryPool(const SimpleMemoryPool &src_pool) = delete;

  SimpleMemoryPool &operator=(const SimpleMemoryPool &src_pool) = delete;

  SimpleMemoryPool(SimpleMemoryPool &&src_pool) = delete;

 private:
  int _decideGranuality(int request_gran);

  struct Block {
    Block(void *src_data, size_t cnt) : data(src_data), byte_sz(cnt) {}
    void *data{nullptr};
    size_t byte_sz{0};
    bool free{true};
  };

  int granuality_{64};
  int byte_count_{0};

  std::mutex mutex_;
  std::vector<Block> blocks_;
  std::unordered_map<int, int> used_buffer_;
  int ref_cnt_{0};

  std::unique_ptr<void *> pool_data_;
};

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

}  // namespace Lolly
