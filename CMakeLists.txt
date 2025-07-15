cmake_minimum_required(VERSION 3.9.0)
include(CheckCXXCompilerFlag)

project(blanczos)
set (BLANCZOS_VERSION       "0.0.1")
set (BLANCZOS_AUTHORS       "Sebastian Wouters")
set (BLANCZOS_LICENSE       "BSD 3-Clause License")
set (BLANCZOS_LIB_SOVERSION 0)

set (CMAKE_VERBOSE_MAKEFILE ON)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug|Release|RelWithDebInfo|MinSizeRel" FORCE)
endif()

check_cxx_compiler_flag (-Wall HAS_WALL)
if (HAS_WALL)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

find_package (OpenMP)
if (OpenMP_CXX_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

check_cxx_compiler_flag (-xHost HAS_XHOST)
check_cxx_compiler_flag (-march=native HAS_MARCH_NATIVE)
if (HAS_XHOST)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
elseif (HAS_MARCH_NATIVE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT HAS_IPO OUTPUT OUTPUT_IPO)
if (HAS_IPO)
    message(STATUS "IPO is supported.")
else()
    message(WARNING "IPO is not supported: ${OUTPUT_IPO}")
endif()

#check_cxx_compiler_flag (-ipo HAS_IPO)
#if (HAS_IPO)
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
#endif()

#check_cxx_compiler_flag (-flto HAS_FLTO)
#if (HAS_FLTO)
#    set (CMAKE_CXX_FLAGS "-flto ${CMAKE_CXX_FLAGS}")
#endif()

set (BLANCZOS_SOURCE_FILES "src/blanczos.cpp")

add_library                (blanczos-base   OBJECT     ${BLANCZOS_SOURCE_FILES})
target_include_directories (blanczos-base   PRIVATE    ${CMAKE_SOURCE_DIR}/src)
set_target_properties      (blanczos-base   PROPERTIES POSITION_INDEPENDENT_CODE 1)
if (HAS_IPO)
    set_property    (TARGET blanczos-base   PROPERTY   INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

add_library                (blanczos-shared SHARED     $<TARGET_OBJECTS:blanczos-base>)
set_target_properties      (blanczos-shared PROPERTIES #SOVERSION ${BLANCZOS_LIB_SOVERSION}
                                                       #MACOSX_RPATH ON
                                                       OUTPUT_NAME "blanczos"
                                                       EXPORT_NAME "blanczos")
if (HAS_IPO)
    set_property    (TARGET blanczos-shared PROPERTY   INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set (BLANCZOS_TESTS "test1" "test2" "test3")

foreach (ITEM ${BLANCZOS_TESTS})
    add_executable             (${ITEM} tests/${ITEM}.cpp)
    target_link_libraries      (${ITEM} blanczos-shared)
    target_include_directories (${ITEM} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    if (HAS_IPO)
        set_property (TARGET ${ITEM} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endforeach()

enable_testing()

add_test(sparse   test1)
add_test(blanczos test2)
add_test(example  test3)

