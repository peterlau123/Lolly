namespace Lolly {

class LRUCache {
public:
  LRUCache(int capacity)
      : capacity_(capacity), cur_length_(0), head_(nullptr), tail_(nullptr) {}

  int get(int key);

  void put(int key, int value);

private:
  int capacity_;
  int cur_length_;

  struct Record;
  Record *head_;
  Record *tail_;

  // return nullptr when list empty or key not exist
  // if list not empty
  // when key exists,increment each record's decay except the kay associated
  // record when key not exists,increment each record's decay
  Record *_check_exist_and_update(int key);

  // ensure that records are ordered by increasing decay
  // least used record is at the tail
  void _reorder_records(Record *ptr);

  void _insert_before(Record *pos, Record *cur);
};
} // namespace Lolly
