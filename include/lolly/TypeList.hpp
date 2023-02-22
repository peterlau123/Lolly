/*
 * Copyright (C) 2021 by Sensetime Group Limited. All rights reserved.
 * Liu Xin <liuxin@sensetime.com>
 */
#pragma once

namespace Lolly {

template <typename... Types> class TypeList {};

template <typename Head, typename... Types> class Front {
  using Type = Head;
};

} // namespace Lolly
