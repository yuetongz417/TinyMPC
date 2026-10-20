# Basic examples
add_executable(quadrotor_tracking quadrotor_tracking.cpp)
target_link_libraries(quadrotor_tracking LINK_PUBLIC tinympcstatic)

add_executable(quadrotor_hovering quadrotor_hovering.cpp)
target_link_libraries(quadrotor_hovering LINK_PUBLIC tinympcstatic)

add_executable(cartpole_example cartpole_example.cpp)
target_link_libraries(cartpole_example LINK_PUBLIC tinympcstatic)

add_executable(cartpole_example_modified cartpole_example_modified.cpp)
target_link_libraries(cartpole_example_modified PRIVATE tinympcstatic)

# Linear constraint example
add_executable(quadrotor_linear_constraints quadrotor_linear_constraints.cpp)
target_link_libraries(quadrotor_linear_constraints LINK_PUBLIC tinympcstatic)

# Time-varying Linear constraint example
add_executable(quadrotor_tv_linear_constraints quadrotor_tv_linear_constraints.cpp)
target_link_libraries(quadrotor_tv_linear_constraints LINK_PUBLIC tinympcstatic)

# Conic example
add_executable(rocket_landing_mpc rocket_landing_mpc.cpp)
target_link_libraries(rocket_landing_mpc LINK_PUBLIC tinympcstatic)

# Codegen examples
add_executable(codegen_random codegen_random.cpp)
target_link_libraries(codegen_random LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(codegen_random LINK_PUBLIC "stdc++fs")
endif(MINGW)

add_executable(codegen_cartpole codegen_cartpole.cpp)
target_link_libraries(codegen_cartpole LINK_PUBLIC tinympcstatic)
if(MINGW)
    target_link_libraries(codegen_cartpole LINK_PUBLIC "stdc++fs")
endif(MINGW)

# Random MPC: ADMM vs reflected Halpern fixed restart
add_executable(random_mpc_toy_benchmark
    ${CMAKE_SOURCE_DIR}/benchmarks/random_mpc/random_mpc_toy_benchmark.cpp
)

target_link_libraries(random_mpc_toy_benchmark
    PRIVATE tinympcstatic
)

target_include_directories(random_mpc_toy_benchmark
    PRIVATE
        "${MPC_BENCH_PROBLEM_DIR}"
        "${CMAKE_SOURCE_DIR}/src/tinympc"
)

# Quadcopter: ADMM vs reflected Halpern restart
add_executable(quadcopter_mpc_benchmark
    ${CMAKE_SOURCE_DIR}/benchmarks/quadcopter/quadcopter_mpc_benchmark.cpp
)

target_link_libraries(quadcopter_mpc_benchmark
    PRIVATE tinympcstatic
)

target_include_directories(quadcopter_mpc_benchmark
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/tinympc"
        "${CMAKE_SOURCE_DIR}/benchmarks/quadcopter"
)

# Append this to TinyMPC/examples/CMakeLists.txt
# Assumes the modified TinyMPC src/tinympc/CMakeLists.txt already compiles
# halpern_reflected.cpp into tinympcstatic (as in the uploaded src.zip).
add_executable(quadcopter_singular_native
    ${CMAKE_SOURCE_DIR}/benchmarks/quadcopter/quadcopter_singular_native.cpp)
target_link_libraries(quadcopter_singular_native PRIVATE tinympcstatic)
target_include_directories(quadcopter_singular_native PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/tinympc
    ${CMAKE_SOURCE_DIR}/benchmarks/quadcopter
    ${CMAKE_SOURCE_DIR}/include/Eigen
)
