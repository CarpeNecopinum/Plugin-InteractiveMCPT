#include "../MonteCuda.hh"
#include "../InfoStructs.hh"

#include <cuda.h>
#include "Cuda/cutil_math.h"
#include <curand_kernel.h>
#include <vector_functions.h>
#include <iostream>

#include "Geometry.hh"
#include "Sampling.hh"
#include "BRDF.hh"

#define CUDA_CHECK cudaCheck(__LINE__);


void cudaCheck(int line)
{
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error %i: %s\n", line, cudaGetErrorString(error));
        exit(-1);
    }
}

/** CUDA Part **/

__device__ float mcBrightness(const float3& color) {
    // Convert to Y (Yuv Color space)
    return (0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
}



template<int RECURSIONS_LEFT>
__device__ float3 mcTrace(mcRay ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount, curandState* state)
{
    mcIntersection hit = mcIntersectScene(ray, mats, geometry, triCount);

    float3 emitted = hit.material.emissiveColor;

    if((hit.depth == FLT_MAX) || (dot(hit.normal, -ray.direction) < 0.f))
        return make_float3(0.0f, 0.0f, 0.0f);

    float diffuseReflectance = mcBrightness(hit.material.diffuseColor);
    float specularReflectance = mcBrightness(hit.material.specularColor);
    float totalReflectance = diffuseReflectance + specularReflectance;

    float3 mirrored = mcMirror(ray.direction, hit.normal);


    float3 sample;
    ((curand_uniform(state) * totalReflectance) <= diffuseReflectance)
        ? sample = mcRandomDirCosTheta(hit.normal, state)
        : sample = mcRandomDirCosPowerTheta(mirrored, hit.material.specularExponent, state);
    float density = (diffuseReflectance / totalReflectance) * mcDensityCosTheta(hit.normal, sample)
                  + (specularReflectance / totalReflectance) * mcDensityCosPowerTheta(mirrored, hit.material.specularExponent, sample);

    mcRay reflectedRay = {hit.position, sample};
    float costheta = dot(sample, hit.normal);

    float3 reflected = mcPhong(hit.material, ray.direction, reflectedRay.direction, hit.normal) *
             mcTrace<RECURSIONS_LEFT-1>(reflectedRay, mats, geometry, triCount, state) * costheta / density;

    return reflected + emitted;
}

template<>
__device__ float3 mcTrace<0>(mcRay ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount, curandState* state)
{
    return make_float3(0.f, 0.f, 0.f);
}


__device__ float cudaRandomSymmetric(curandState* state) { return curand_uniform(state) * 2.f - 1.f; }

__global__ void tracePixels(QueuedPixel* pixels, float3* output, mcMaterial* mats, mcTriangle* geometry, mcCameraInfo* cam, size_t triCount, RenderSettings settings, uint32_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    QueuedPixel& coord = pixels[idx];

    if (coord.x == -1) return;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float x = float(coord.x) + .5f * cudaRandomSymmetric(&state);
    float y = float(coord.y) + .5f * cudaRandomSymmetric(&state);

    /* Ray Setup */
    float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
    mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

    /* Actual Path Tracing */
    output[idx] = mcTrace<4>(ray, mats, geometry, triCount, &state);
}

/** C(++) Part **/

mcTriangle*  devTriangles = 0;
size_t devTriangleCount = 0;
mcMaterial*  devMaterials = 0;
mcCameraInfo*   devCamera = 0;

void uploadBuffers(mcMaterial *materials, size_t materialCount, mcTriangle *tris, size_t triCount)
{
    cudaFree(devMaterials); CUDA_CHECK
    cudaMalloc(&devMaterials, sizeof(mcMaterial) * materialCount); CUDA_CHECK
    cudaMemcpy(devMaterials, materials, sizeof(mcMaterial) * materialCount, cudaMemcpyHostToDevice); CUDA_CHECK

    cudaFree(devTriangles); CUDA_CHECK
    cudaMalloc(&devTriangles, sizeof(mcTriangle) * triCount); CUDA_CHECK
    cudaMemcpy(devTriangles, tris, sizeof(mcTriangle) * triCount, cudaMemcpyHostToDevice); CUDA_CHECK

    devTriangleCount = triCount;
}

void uploadCameraInfo(const CameraInfo& cam)
{
    mcCameraInfo dev;
    dev.eye_point = toCudaVec(cam.eye_point);
    dev.image_plane_start = toCudaVec(cam.image_plane_start);
    dev.x_dir = toCudaVec(cam.x_dir);
    dev.y_dir = toCudaVec(cam.y_dir);

    cudaFree(devCamera); CUDA_CHECK
    cudaMalloc(&devCamera, sizeof(mcCameraInfo)); CUDA_CHECK
    cudaMemcpy(devCamera, &dev, sizeof(mcCameraInfo), cudaMemcpyHostToDevice); CUDA_CHECK
}


void cudaTracePixels(std::vector<QueuedPixel> &pixels, RenderSettings settings, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    QueuedPixel* devPixels;
    cudaMalloc(&devPixels, sizeof(QueuedPixel) * pixels.size()); CUDA_CHECK
    cudaMemcpy(devPixels, pixels.data(), sizeof(QueuedPixel) * pixels.size(), cudaMemcpyHostToDevice); CUDA_CHECK

    float3* devResults;
    cudaMalloc(&devResults, sizeof(float3) * pixels.size()); CUDA_CHECK

    assert(pixels.size() % CUDA_BLOCK_SIZE == 0);

    tracePixels<<<pixels.size() / cudaBlockSize(), cudaBlockSize()>>>(devPixels, devResults, devMaterials, devTriangles, devCamera, devTriangleCount, settings, rand() << 16 | rand()); CUDA_CHECK

    float3* hostResults = new float3[pixels.size()];
    cudaMemcpy(hostResults, devResults, sizeof(float3) * pixels.size(), cudaMemcpyDeviceToHost); CUDA_CHECK

    cudaFree(devPixels); CUDA_CHECK
    cudaFree(devResults); CUDA_CHECK

    for (size_t i = 0; i < pixels.size(); ++i)
    {
        QueuedPixel& pixel = pixels[i];
        size_t index = pixel.x + pixel.y * imageWidth;
        colorMap[index] += toACG3(hostResults[i]);
        sampleCounter[index]++;
    }

    delete[] hostResults;
}

