#include "lolly/container/LRUCache.hpp"

namespace Lolly {

// TODO(liuxin):better use smart pointer
struct LRUCache::Record {
  Record() : key(-1), val(-1), decay(0), next(nullptr), prev(nullptr) {}

  Record(int key_, int val_)
      : key(key_), val(val_), decay(0), next(nullptr), prev(nullptr) {}

  void IncrementDecay(int64_t decay_ = 1) { decay += decay_; }

  void DecrementDecay(int64_t decay_ = 1) { decay -= decay_; }

  int key;
  int val;
  int64_t decay; // the smaller ,the more recently used
  Record *next;
  Record *prtev;
};

int LRUCache::get(int key) {
  auto *result = _check_exist(key);
  if (nullptr != result) {
    result->DecrementDecay(1);
    _reorder_records(result);
  }
  return result;
}

void LRUCache::put(int key, int value) {
  if (nullptr == records_) {
    head_ = new Record(key, value);
    head_->next = nullptr;
    head_->prev = nullptr;
    // head_->decay--;
    tail_ = head_;
    cur_length_++;
  } else {
    Record *cur = _check_exist_and_update(key);
    if (nullptr == cur) {
      auto *tmp = tail;
      auto *prev = tmp->prev;
      if (capacity_ <= cur_length_) {
        delete tmp;
      }
      tail = new Record(key, value);
      prev->next = tail;
      tail->prev = prev;
      // tail->decay--;
    } else {
      cur->val = value;
      // cur->decay--;
      cur->DecrementDecay(1);
      _reorder_records(cur);
    }
  }
}

Record *LRUCache::_check_exist_and_update(int key) {
  Record *result = nullptr;
  Record *tmp = head_;
  while (nullptr != tmp) {
    if (key == tmp->key) {
      result = tmp;
      break;
    } else {
      tmp->IncrementDecay(1);
      tmp = tmp->next;
    }
  }
  return result;
}

// ensure that records are ordered by increasing decay
// least used record is at the tail
void LRUCache::_reorder_records(Record *ptr) {
  if (nullptr == records_) {
    return;
  }
  auto *tmp = ptr;
  while (nullptr != tmp) {
    Record *prev = tmp->prev;
    if (tmp->decay < prev->decay) {
      _insert_before(prev, tmp);
    }
    tmp = tmp->prev;
  }
}

void LRUCache::_insert_before(Record *pos, Record *cur) {
  cur->prev->next = cur->next;
  cur->next->prev = cur->prev;
  cur->next = pos;
  cur->prev = pos->prev;
  pos->prev = cur;
  cur->prev->next = cur;
}

} // namespace Lolly
