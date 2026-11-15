add_library(tyche_beliefs INTERFACE)
add_library(tyche::beliefs ALIAS tyche_beliefs)
target_include_directories(tyche_beliefs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tyche_beliefs INTERFACE cxx_std_20)
# The Möbius transforms on the lattice of subsets are morphe's.
target_link_libraries(tyche_beliefs INTERFACE morphe::combinatorics)
if(TYCHE_CHECKS)
  # The credal set's lower expectation is checked against a linear programme on calx's simplex method.
  tyche_add_check(check_tyche_belief_functions SOURCES tests/check_belief_functions.cpp LIBRARIES tyche::beliefs calx::random calx::optimisation)
endif()
