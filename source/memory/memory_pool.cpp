#include "lolly/memory/MemoryPool.hpp"

#include <iostream>
#include <memory>

namespace Lolly {

// Choose the proper  granuality
// align address
int SimpleMemoryPool::_decideGranuality(int request_gran) {
  static int default_granuality = 64;
  static int max_granuality = 1024;

  if (default_granuality == request_gran) {
    return granuality_;
  }

  // Eval granuality waste
  float waste_ratio = 1.0f;
  auto cal_waste_ratio = [](int dst_gran, int request_gran) -> float {
    return (dst_gran - request_gran) * 1.0f / dst_gran;
  };

  // cal final granuality
  int result_granuality = default_granuality;

  if (result_granuality < request_gran) {
    while ((result_granuality << 1) <= request_gran) {
      result_granuality = result_granuality << 1;
    }

  } else {
    while (request_gran <= (result_granuality >> 1)) {
      result_granuality = result_granuality >> 1;
    }
  }

  if (max_granuality < result_granuality) {
    std::cout << "Request granuzality exceeds max granuality allowed!"
              << std::endl;
    result_granuality = max_granuality;
  }

  std::cout << "Final granuality " << result_granuality << " with waste ratio "
            << cal_waste_ratio(result_granuality, request_gran);

  return result_granuality;
}

SimpleMemoryPool::SimpleMemoryPool(size_t byte_count, int granuality = 64)
    : granuality_(64), byte_count_(0) {
  // Determine space characteristics
  int final_granuality = this->_decideGranuality(granuality);
  int final_block_num = (byte_count + final_granuality - 1) / final_granuality;
  int final_byte_count = final_block_num * final_granuality;

  granuality_ = final_granuality;
  byte_count_ = final_byte_count;

  // allocate space
  blocks_.resize(final_block_num);

  uint8_t *data = (uint8_t *)malloc(byte_count_);
  if (nullptr == data) {
    std::cerr << "Construct memory pool failed" << std::endl;
  }

  // TODO:consider alignment?

  Deleter deleter;
  pool_data_ = std::unique_ptr<uint8_t, Deleter>(data, deleter);

  for (int index = 0; index < final_block_num; index++) {
    Block block(data + index * final_granuality, granuality_);
    blocks_.push_back(block);
  }
}

void *SimpleMemoryPool::Get(size_t request_sz) {
  if (byte_count_ < request_sz) {
    std::cerr << "Request size exceeds memory pool size" << std::endl;
    return nullptr;
  }
  int need_block_num = (request_sz + granuality_ - 1) / granuality_;

  std::lock_guard<std::mutex> guard(mutex_);

  int left_block_num = blocks_.size() - ref_cnt_;
  if (left_block_num < need_block_num) {
    std::cerr << "Not enough block for the request size" << std::endl;
    return nullptr;
  }

  // check continuous free block number
  int continuous_number = 0;
  int start_block_idx = -1;
  for (size_t i = 0; 0 < blocks_.size(); i++) {
    if (blocks_[i].free) {
      continuous_number++;
      if (-1 == start_block_idx) {
        start_block_idx = i;
      }
    } else {
      continuous_number = 0;
      start_block_idx = -1;
    }
  }

  if (0 == continuous_number || -1 == start_block_idx ||
      continuous_number < need_block_num) {
    std::cerr
        << "Cannot find enough continuous free blocks for the request size"
        << std::endl;
    return nullptr;
  }

  for (int idx = start_block_idx; idx < start_block_idx + continuous_number;
       idx++) {
    blocks_[idx].free = false;
    ref_cnt_++;
  }
  void *found_data = blocks_[start_block_idx].data;
  used_buffer_[start_block_idx] = continuous_number;

  return found_data;
}

void SimpleMemoryPool::Put(void *data) {
  if (nullptr == data)
    return;

  std::lock_guard<std::mutex> guard(mutex_);

  int result_idx = -1;
  for (size_t i = 0; i < blocks_.size(); i++) {
    if (data == blocks_[i].data) {
      result_idx = i;
      break;
    }
  }
  if (-1 == result_idx) {
    std::cerr << "Illegal free memory not owned by memory pool" << std::endl;
  }

  int continual_number = used_buffer_[result_idx];
  for (int start_idx = result_idx; start_idx < result_idx + continual_number;
       start_idx++) {
    blocks_[start_idx].free = true;
    ref_cnt_--;
  }
  used_buffer_.erase(result_idx);
}

SimpleMemoryPool::~SimpleMemoryPool() {
  pool_data_.reset();
  blocks_.clear();
  used_buffer_.clear();
  ref_cnt_ = 0;
}

} // namespace Lolly
