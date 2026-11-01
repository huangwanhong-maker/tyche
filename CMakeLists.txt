# tyche: chance — probability, statistics and inference.
#
# Every module is a target tyche::<module>. tyche uses calx's, morphe's and physis's targets:
# inside The Science Infrastructures, the siblings foundations/calx, foundations/morphe and
# foundations/physis (added here when no larger project has added them); built alone elsewhere,
# they would be fetched at pinned versions (not yet written). Built on its own, tyche also builds
# its checks (TYCHE_CHECKS), programs on calx's verification registered with CTest.

cmake_minimum_required(VERSION 3.20)
project(tyche VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(TYCHE_TOP_LEVEL ON)
else()
  set(TYCHE_TOP_LEVEL OFF)
endif()
option(TYCHE_CHECKS "Build tyche's checks" ${TYCHE_TOP_LEVEL})

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT TARGET calx::linalg)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../calx/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../calx ${CMAKE_CURRENT_BINARY_DIR}/calx)
  else()
    message(FATAL_ERROR "tyche needs calx: build it inside The Science Infrastructures (foundations/calx).")
  endif()
endif()
if(NOT TARGET morphe::topology)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../morphe/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../morphe ${CMAKE_CURRENT_BINARY_DIR}/morphe)
  else()
    message(FATAL_ERROR "tyche needs morphe: build it inside The Science Infrastructures (foundations/morphe).")
  endif()
endif()
if(NOT TARGET physis::units)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../physis/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../physis ${CMAKE_CURRENT_BINARY_DIR}/physis)
  else()
    message(FATAL_ERROR "tyche needs physis: build it inside The Science Infrastructures (foundations/physis).")
  endif()
endif()
if(TYCHE_CHECKS)
  enable_testing()
endif()

# tyche_add_check(<name> SOURCES <files> LIBRARIES <targets>): calx_add_check, knowing where the
# calling module's reference values are (TYCHE_FIXTURES = <module>/tests/fixtures).
function(tyche_add_check name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})
  calx_add_check(${name} SOURCES ${ARG_SOURCES} LIBRARIES ${ARG_LIBRARIES})
  target_compile_definitions(${name} PRIVATE TYCHE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
endfunction()

# The modules in their order (decision 0003).
add_subdirectory(probability)
add_subdirectory(descriptive)
add_subdirectory(regression)
add_subdirectory(multivariate)
add_subdirectory(time_series)
add_subdirectory(spatial)
add_subdirectory(testing)
add_subdirectory(learning)
