cmake_minimum_required(VERSION 3.15)
project(matvec-prod CXX)

include(cmake/standard_project_settings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Link this 'library' to use the warnings specified in compiler_warnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/compiler_warnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/sanitizers.cmake)
enable_sanitizers(project_options)

# allow for static analysis options
include(cmake/static_analyzers.cmake)

# Conan dependencies
set(CONAN_REQUIRES
    fmt/6.1.2
    benchmark/1.5.0
    eigen/3.3.7
    blaze/3.7
)
set(CONAN_OPTIONS)

include(cmake/conan.cmake)
run_conan()

include_directories(include)

add_library(simd INTERFACE)
target_link_libraries(
  simd
  INTERFACE project_options
            project_warnings
            CONAN_PKG::fmt
            CONAN_PKG::benchmark
            CONAN_PKG::eigen
            CONAN_PKG::blaze
)

add_library(extern test/bench.cpp)
add_executable(check test/check_result.cpp)
target_link_libraries(check extern)

# Parallelize extern build
foreach(float_type f32 f64)
  foreach(n_cols 2 4 8)
    set(extern_target extern_${float_type}_${n_cols})
    add_library(${extern_target} test/extern.cpp)
    target_link_libraries(${extern_target} simd)
    target_compile_definitions(
      ${extern_target} PRIVATE FLOAT_TYPE=${float_type} NCOLS=${n_cols}
    )
    target_link_libraries(extern ${extern_target})
  endforeach(n_cols)
endforeach(float_type)

foreach(float_type f32 f64)
  foreach(n_cols 2 4 8)
    # Multiple benchmarks [17, 128]
    set(target ${float_type}_${n_cols})
    add_executable(${target} test/multi_bench.cpp)
    target_link_libraries(${target} simd extern)
    target_compile_definitions(
      ${target}
      PRIVATE FLOAT_TYPE=${float_type}
              NCOLS=${n_cols}
              BM_EIGEN
              BM_BLAZE
              BM_SIMD
    )
    # Individual benchmarks [0, 16]
    foreach(n_rows RANGE 16)
      foreach(method EIGEN BLAZE SIMD)
        set(target ${float_type}_${n_cols}_${n_rows}_${method})
        add_executable(${target} test/single_bench.cpp)
        target_link_libraries(${target} simd extern)
        target_compile_definitions(
          ${target}
          PRIVATE FLOAT_TYPE=${float_type}
                  NCOLS=${n_cols}
                  NROWS=${n_rows}
                  BM_${method}
        )
      endforeach(method)
    endforeach(n_rows)
  endforeach(n_cols)
endforeach(float_type)
