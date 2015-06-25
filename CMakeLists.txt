include (plugin)

find_package(CXX11)

openflipper_plugin (CFLAGS_ADD "${CXX11_FLAGS} -ffast-math")
