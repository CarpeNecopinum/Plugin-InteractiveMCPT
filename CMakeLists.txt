include (plugin)

find_package(CXX11 REQUIRED)

# --------------------------------------------
# If you don't want to use CUDA, set this to 0
set(USE_CUDA 1)
# --------------------------------------------


if(USE_CUDA)
openflipper_plugin (
    CFLAGSADD "${CXX11_FLAGS} -g -ffast-math"
    DEPS CUDA
)
else()
openflipper_plugin (
    CFLAGSADD "${CXX11_FLAGS} -g -ffast-math"
)
endif()

if(USE_CUDA)
message("YAY Cuda!")
add_definitions(-DHAS_CUDA)
else()
message("No Cuda! You suck!")
endif()

add_definitions(-DCUDA_BLOCK_SIZE=128)
