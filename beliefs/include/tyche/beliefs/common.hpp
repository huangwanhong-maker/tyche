// What tyche::beliefs's headers share: subsets of a finite frame as bit masks, a frame with named elements, and the
// refusal (decision 0023).
//
// A frame of discernment Ω = {ω₀, …, ω_{n−1}} (Shafer 1976, A Mathematical Theory of Evidence, ch. 2) holds the
// mutually exclusive answers to one question. A subset A ⊆ Ω is a bit mask: ω_i ∈ A exactly when bit i of A is
// set; ∅ is 0 and Ω is 2ⁿ − 1. Frames of up to 30 elements are held (max_frame); the operations that tabulate all
// 2ⁿ subsets (the dense transforms, membership of the credal set) are refused above 20 (dense_frame).

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyche::beliefs {

class Refusal : public std::invalid_argument {
 public:
  explicit Refusal(const std::string& reason) : std::invalid_argument("tyche::beliefs: " + reason) {}
};

using Set = std::uint32_t;

inline constexpr std::size_t max_frame = 30;
inline constexpr std::size_t dense_frame = 20;

namespace detail {
inline void require(bool ok, const std::string& what) {
  if (!ok) throw Refusal(what);
}
inline void check_frame(std::size_t frame) {
  require(frame >= 1 && frame <= max_frame, "a frame has between 1 and " + std::to_string(max_frame) + " elements, not " + std::to_string(frame));
}
inline void check_dense(std::size_t frame) {
  require(frame <= dense_frame, "the 2ⁿ subsets of a frame of " + std::to_string(frame) + " elements are not tabulated (at most " +
                                    std::to_string(dense_frame) + ")");
}
}  // namespace detail

// Ω, the frame itself.
inline Set full(std::size_t frame) {
  detail::check_frame(frame);
  return (Set{1} << frame) - 1;
}
inline Set singleton(std::size_t i) {
  detail::require(i < max_frame, "an element out of range");
  return Set{1} << i;
}
inline std::size_t cardinality(Set A) { return static_cast<std::size_t>(std::popcount(A)); }
inline Set complement(Set A, std::size_t frame) { return full(frame) & ~A; }
inline bool within(Set A, std::size_t frame) { return (A & ~full(frame)) == 0; }

// A frame whose elements have names.
struct Frame {
  std::vector<std::string> elements;

  std::size_t size() const { return elements.size(); }
  Set all() const { return full(size()); }
  std::size_t index(const std::string& name) const {
    for (std::size_t i = 0; i < elements.size(); i++)
      if (elements[i] == name) return i;
    throw Refusal("the frame has no element '" + name + "'");
  }
  Set set(const std::vector<std::string>& names) const {
    Set A = 0;
    for (const auto& name : names) A |= Set{1} << index(name);
    return A;
  }
  std::vector<std::string> names(Set A) const {
    detail::require(within(A, size()), "a set outside the frame");
    std::vector<std::string> out;
    for (std::size_t i = 0; i < size(); i++)
      if (A >> i & 1) out.push_back(elements[i]);
    return out;
  }
};

inline Frame frame(std::vector<std::string> elements) {
  detail::check_frame(elements.size());
  for (std::size_t i = 0; i < elements.size(); i++) {
    detail::require(!elements[i].empty(), "an element of the frame has no name");
    for (std::size_t j = i + 1; j < elements.size(); j++) detail::require(elements[i] != elements[j], "two elements of the frame are named '" + elements[i] + "'");
  }
  return Frame{std::move(elements)};
}

}  // namespace tyche::beliefs
