#include <gtest/gtest.h>
#include <lolly/parallel/basic.h>
#include <vector>

TEST(Lolly.parallel.basic, reduce) {
  using namespace Lolly::parallel;

  const max_seq_len = 1000;
  std::vector<float> input;
  for (int i = 1; i <= max_seq_len; i++) {
    input.emplace_back(i);
  }
  float *output = new int(0);
  ReduceType reduce_type = Reduce::SUM;
  Lolly::parallel::reduce(static_cast<float *>(data()), &out, max_seq_len,
                          reduce_type);
  EXPECT_EQ(*output, 250250);
}