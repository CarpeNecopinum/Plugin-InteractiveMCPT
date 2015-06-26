include (plugin)

find_package(CXX11 REQUIRED)

openflipper_plugin (CFLAGSADD "${CXX11_FLAGS} -ffast-math")
