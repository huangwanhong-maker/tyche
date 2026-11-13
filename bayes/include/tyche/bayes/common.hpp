// What tyche::bayes's headers share.
#pragma once

#include <stdexcept>
#include <string>

namespace tyche::bayes::detail {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::bayes: ") + what);
}
}  // namespace tyche::bayes::detail
