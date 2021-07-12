set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../libs)

#set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -O3")
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

#lib dir
link_directories("../libs")

#libgc
add_library(gc SHARED "gc/gc.cpp")
target_link_libraries(gc ${RELIC_LIBRARIES}   
                         ${EMP-OT_LIBRARIES}  ${OPENSSL_LIBRARIES}  
                         ${Boost_LIBRARIES}   ${GMP_LIBRARIES})

#lib ppml
file(GLOB ppmlsrc "base/*.cpp" "ml/*.cpp")
add_library(ppml SHARED ${ppmlsrc})
target_link_libraries(ppml  ${TORCH_LIBRARIES} "gc")

#test_triplets
add_executable(test_triplets "test/test_triplets.cpp")
target_link_libraries(test_triplets "gc" "ppml" "paillier")

#test_func
add_executable(test_func "test/test_func.cpp")
target_link_libraries(test_func "gc" "ppml" "paillier")

#data_process
add_executable(data_process "utils/dataprocess.cpp")
target_link_libraries(data_process "gc" "ppml" "paillier")

#bin_mnist_linear
add_executable(bin_mnistlinear "train/linear/bin_mnistlinear.cpp")
target_link_libraries(bin_mnistlinear "gc" "ppml" "paillier")

#bin_mnistlog
add_executable(bin_mnistlog "train/logistic/bin_mnistlog.cpp")
target_link_libraries(bin_mnistlog "gc" "ppml" "paillier")

#bin_gisettelog
add_executable(gisettelog "train/logistic/gisettelog.cpp")
target_link_libraries(gisettelog "gc" "ppml" "paillier")

#mnistlog
add_executable(mnistlog "train/logistic/mnistlog.cpp")
target_link_libraries(mnistlog "gc" "ppml" "paillier")

#mnistnet
add_executable(mnistnet "train/networks/mnistnet.cpp")
target_link_libraries(mnistnet "gc" "ppml" "paillier")

#quotient
add_executable(quotient "train/networks/quotient.cpp")
target_link_libraries(quotient "gc" "ppml" "paillier")

#mnistconv
add_executable(mnistconv "train/convnetworks/mnistconv.cpp")
target_link_libraries(mnistconv "gc" "ppml" "paillier")

#cifarconv
add_executable(cifarconv "train/convnetworks/cifarconv.cpp")
target_link_libraries(cifarconv "gc" "ppml" "paillier")