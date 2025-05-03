#include <lolly/version.h>
#include <string>

TEST("Lolly","version") {
  static_assert(std::string_view(LOLLY_VERSION) == std::string_view("0.1.0"));
  EXPECT_TRUE(std::string(GREETER_VERSION) == std::string("0.1.0"));
}

