# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
<<<<<<< HEAD
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(proto)
=======
add_library(caffeproto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
caffe_default_properties(caffeproto)
target_link_libraries(caffeproto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(caffeproto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Caffe_LINKER_LIBS 0 PUBLIC caffeproto) # note, crucial to prepend!
>>>>>>> caffe-bvlc-dev/master

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${PROJECT_SOURCE_DIR})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs})
<<<<<<< HEAD
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
caffe_default_properties(caffe)
=======
caffe_default_properties(caffe)
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
target_include_directories(caffe ${Caffe_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${Caffe_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)
target_compile_definitions(caffe ${Caffe_DEFINITIONS})
if(Caffe_COMPILE_OPTIONS)
  target_compile_options(caffe ${Caffe_COMPILE_OPTIONS})
endif()
>>>>>>> caffe-bvlc-dev/master
set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    )

# ---[ Tests
 add_subdirectory(test)

# ---[ Install
<<<<<<< HEAD
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION include)
install(FILES ${proto_hdrs} DESTINATION include/caffe/proto)
install(TARGETS caffe proto EXPORT CaffeTargets DESTINATION lib)
=======
install(DIRECTORY ${Caffe_INCLUDE_DIR}/caffe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${proto_hdrs} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/caffe/proto)
install(TARGETS caffe caffeproto EXPORT CaffeTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
>>>>>>> caffe-bvlc-dev/master

file(WRITE ${PROJECT_BINARY_DIR}/__init__.py)
list(APPEND proto_python ${PROJECT_BINARY_DIR}/__init__.py)
install(PROGRAMS ${proto_python} DESTINATION python/caffe/proto)

<<<<<<< HEAD

=======
>>>>>>> caffe-bvlc-dev/master
