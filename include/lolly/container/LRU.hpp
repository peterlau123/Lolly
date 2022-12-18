class LRUCache {
public:
  LRUCache(int capacity)
      : capacity_(capacity), cur_length_(0), head_(nullptr), tail_(nullptr) {}

  int get(int key) {
    auto *result = check_exist(key);
    if (nullptr != result) {
      result->decay--;
      reorder_records(head);
    }
    return result;
  }

  void put(int key, int value) {
    if (nullptr == records_) {
      head_ = new Record(key, value);
      head_->next = nullptr;
      head_->prev = nullptr;
      head_->decay--;

      tail_ = head_;

      cur_length_++;
    } else {
      Record *cur = check_exist(key);
      if (nullptr == cur) {
        auto *tmp = tail;
        auto *prev = tmp->prev;
        if (capacity_ <= cur_length_) {
          delete tmp;
        }
        tail = new Record(key, value);
        prev->next = tail;
        tail->prev = prev;
        tail->decay--;
      } else {
        cur->val = value;
        cur->decay--;
        reorder_records(cur);
      }
    }
  }

private:
  int capacity_;
  int cur_length_;

  struct Record {
    Record()
        : key(key_), val(val_), decay(std::numeric_limits<int>::max()),
          next(nullptr), prev(nullptr) {}
    Record(int key_, int val_)
        : key(key_), val(val_), decay(std::numeric_limits<int>::max()),
          next(nullptr), prev(nullptr) {}
    int key;
    int val;
    int decay; // the smaller ,the more recently used
    Record *next;
    Record *prtev;
  };

  Record *head_;
  Record *tail_;

  // return nullptr when not exist
  // if list not null,increment each decay
  Record *check_exist(int key) {
    Record *result = nullptr;
    Record *tmp = head_;
    while (nullptr != tmp) {
      if (key == tmp->key) {
        result = tmp;
        break;
      } else {
        tmp = tmp->next;
      }
    }
    return result;
  }

  // ensure that records are ordered by increasing decay
  // least used record is at the tail
  void reorder_records(Record *ptr) {
    if (nullptr == records_)
      return;
    auto *tmp = ptr;
    while (nullptr != tmp) {
      Record *prev = tmp->prev;
      if (tmp->decay < prev->decay) {
        insert_before(prev, tmp);
        break;
      }
      tmp = tmp->prev;
    }
  }

  void insert_before(Record *pos, Record *cur) {
    cur->prev->next = cur->next;
    cur->next->prev = cur->prev;
    cur->next = pos;
    cur->prev = pos->prev;
    pos->prev = cur;
    cur->prev->next = cur;
  }
};