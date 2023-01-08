#pragma once

namespace Lolly {

/*
 *@class List
 *
 *List implemented by loop double-linked list
 *Thread-safe
 *
 * */
template <typename T, typename Allocator> class List {
public:
  List(size_t size);

private:
  struct Node;

  Node *nodes_;
};

} // namespace Lolly
