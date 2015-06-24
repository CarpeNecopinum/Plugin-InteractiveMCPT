include (plugin)

find_package(CXX11)

openflipper_plugin (DIRS ${CMAKE_SOURCE_DIR}/Plugin-InteractiveMCPT CFLAGS_ADD "${CXX11_FLAGS} -ffast-math")
