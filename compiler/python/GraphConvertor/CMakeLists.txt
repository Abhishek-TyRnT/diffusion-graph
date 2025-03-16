
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/weightBuffers.proto)
message(${PROTO_FILE})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

# Generate Python code from Proto file
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/weightBuffers.py
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --python_out=${CMAKE_CURRENT_BINARY_DIR}
         --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating Python protocol buffer code"
    VERBATIM
)

add_library(GraphWriter GraphWriter.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(GraphWriter ${Protobuf_LIBRARIES})
target_include_directories(GraphWriter PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(generate_python_proto ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/weightBuffers.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/weightBuffers.py DESTINATION lib/python)

# Add a custom target to copy Python files to the build directory
add_custom_target(copy_python 
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ReadWeights.py ${CMAKE_CURRENT_BINARY_DIR}/ReadWeights.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ReadWeights.py
    COMMENT "Copying Python script to build directory"
)