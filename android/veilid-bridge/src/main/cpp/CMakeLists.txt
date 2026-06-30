cmake_minimum_required(VERSION 3.22.1)
project(optioignon_veilid LANGUAGES CXX)

# SKELETON: builds the JNI stub only. Host-side, link the real veilid-core
# library (static or shared) into this target and replace the stub bodies.
add_library(optioignon_veilid SHARED veilid_bridge.cpp)

find_library(log-lib log)
target_link_libraries(optioignon_veilid ${log-lib})
