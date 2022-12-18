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
  Record *check_exist(int key);

  // ensure that records are ordered by increasing decay
  // least used record is at the tail
  void reorder_records(Record *ptr);

  void insert_before(Record *pos, Record *cur);
};
} // namespace Lolly
