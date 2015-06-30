#include "../MonteCuda.hh"
#include "../InfoStructs.hh"

#include <cuda.h>
#include <iostream>


/** CUDA Part **/

__global__ void fillWithOne(int* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 1;
}

__global__ void tracePixels(Point* pixels, float* output, mcMaterial* mats, mcTriangle* geometry, mcCameraInfo* cam)
{

}


/** C(++) Part **/

mcTriangle* devTriangles = 0;
mcMaterial* devMaterials = 0;
mcCameraInfo*   devCamera = 0;

void cudaTest(void)
{
    const size_t length = 64;
    int data[length];

    int* devPtr;
    cudaMalloc(&devPtr, length * sizeof(int));

    fillWithOne<<<4, 16>>>(devPtr);

    cudaMemcpy(data, devPtr, sizeof(int) * length, cudaMemcpyDeviceToHost);

    for (size_t i = 0u; i < length; i++)
        std::cout << data[i] << std::endl;
}

void uploadBuffers(mcMaterial *materials, size_t materialCount, mcTriangle *tris, size_t triCount)
{
    cudaFree(devMaterials);
    cudaMalloc(&devMaterials, sizeof(mcMaterial) * materialCount);
    cudaMemcpy(materials, devMaterials, sizeof(mcMaterial) * materialCount, cudaMemcpyHostToDevice);

    cudaFree(devTriangles);
    cudaMalloc(&devTriangles, sizeof(mcTriangle) * triCount);
    cudaMemcpy(tris, devTriangles, sizeof(mcTriangle) * triCount, cudaMemcpyHostToDevice);
}

void uploadCameraInfo(const CameraInfo& cam)
{

    mcCameraInfo dev;
    for (int i = 0; i < 3; i++)
    {
        dev.eye_point[i] = (float) cam.eye_point[i];
        dev.image_plane_start[i] = (float) cam.image_plane_start[i];
        dev.x_dir[i] = (float) cam.x_dir[i];
        dev.y_dir[i] = (float) cam.y_dir[i];
    }

    cudaFree(devCamera);
    cudaMalloc(&devCamera, sizeof(mcCameraInfo));
    cudaMemcpy(&dev, devCamera, sizeof(mcCameraInfo), cudaMemcpyHostToDevice);
}


void cudaTracePixels(std::vector<Point> &pixels, RenderSettings settings, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    Point* devPixels;
    cudaMalloc(&devPixels, sizeof(Point) * pixels.size());
    cudaMemcpy(pixels.data(), devPixels, sizeof(Point) * pixels.size(), cudaMemcpyHostToDevice);

    float* devResults;
    cudaMalloc(&devResults, sizeof(float) * 3 * pixels.size());

    tracePixels<<<pixels.size(),1>>>(devPixels, devResults, devMaterials, devTriangles, devCamera);
}

