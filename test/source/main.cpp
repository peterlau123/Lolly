#include <gtest/gtest.h>

// refer to http://google.github.io/googletest/primer.html
// for more usage information
int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
