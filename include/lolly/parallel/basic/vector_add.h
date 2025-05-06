#pragma once

#include "../../utility.h"

namespace Lolly {
namespace parallel {
static void vector_add(_IN_ float *input1, _IN_ float *input2,
                       _INOUT_ float **out, _IN_ int len);
}
} // namespace Lolly