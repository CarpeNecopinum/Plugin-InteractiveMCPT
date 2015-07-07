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

#include "KdTree.hh"

#define CUDA_CHECK
//cudaCheck(__LINE__);


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
             mcTrace(reflectedRay, mats, geometry, triCount, state) * costheta / density;

    return reflected + emitted;
}


template<int RECURSIONS_LEFT>
__device__ float3 mcTrace(mcRay ray, mcMaterial* mats, KdTree<TREE_DEPTH>* tree, curandState* state)
{
    mcIntersection hit = mcIntersectKdTree(ray, mats, tree);


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
             mcTrace<RECURSIONS_LEFT-1>(reflectedRay, mats, tree, state) * costheta / density;

    return reflected + emitted;
}

template<>
__device__ float3 mcTrace<0>(mcRay ray, mcMaterial* mats, KdTree<TREE_DEPTH>* tree, curandState* state)
{
    return make_float3(0.f, 0.f, 0.f);
}

template<int RECURSIONS_LEFT>
__device__ float3 mcTrace(mcRay ray, mcMaterial* mats, mcTriangle* tris, size_t triCount, curandState* state)
{
    mcIntersection hit = mcIntersectScene(ray, mats, tris, triCount);


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
             mcTrace<RECURSIONS_LEFT-1>(reflectedRay, mats, tris, triCount, state) * costheta / density;

    return reflected + emitted;
}

template<>
__device__ float3 mcTrace<0>(mcRay, mcMaterial*, mcTriangle*, size_t, curandState*)
{
    return make_float3(0.f, 0.f, 0.f);
}

__device__ float cudaRandomSymmetric(curandState* state) { return curand_uniform(state) * 2.f - 1.f; }

__global__ void tracePixels(QueuedPixel* pixels, float3* output, mcMaterial* mats, KdTree<TREE_DEPTH>* tree, mcCameraInfo* cam, uint32_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    QueuedPixel& pixel = pixels[idx];

    if (pixel.x == -1) return;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float3 color = make_float3(0.f, 0.f, 0.f);

    const int samples = min(pixel.samples, 10);
    pixel.samples -= samples;
    for (int i = 0; i < samples; i++)
    {
        float x = float(pixel.x) + .5f * cudaRandomSymmetric(&state);
        float y = float(pixel.y) + .5f * cudaRandomSymmetric(&state);

        /* Ray Setup */
        float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
        mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

        /* Actual Path Tracing */
        color += mcTrace<4>(ray, mats, tree, &state);
    }
    output[idx] = color;
}
__global__ void tracePixels(QueuedPixel* pixels, float3* output, mcMaterial* mats, mcTriangle* tris, size_t triCount, mcCameraInfo* cam, uint32_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    QueuedPixel& pixel = pixels[idx];

    if (pixel.x == -1) return;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float3 color = make_float3(0.f, 0.f, 0.f);

    const int samples = min(pixel.samples, 10);
    pixel.samples -= samples;
    for (int i = 0; i < samples; i++)
    {
        float x = float(pixel.x) + .5f * cudaRandomSymmetric(&state);
        float y = float(pixel.y) + .5f * cudaRandomSymmetric(&state);

        /* Ray Setup */
        float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
        mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

        /* Actual Path Tracing */
        color += mcTrace<4>(ray, mats, tris, triCount, &state);
    }
    output[idx] = color;
}

/** C(++) Part **/

mcTriangle*  devTriangles = 0;
size_t devTriangleCount = 0;
mcMaterial*  devMaterials = 0;
mcCameraInfo*   devCamera = 0;

KdTree<TREE_DEPTH>* devKdTree = 0;

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

void uploadKdTree(mcMaterial *materials, size_t materialCount, const std::vector<mcTriangle>& triangles)
{
    cudaFree(devKdTree);
    cudaMalloc(&devKdTree, sizeof(KdTree<TREE_DEPTH>));

    KdTree<TREE_DEPTH> hostTree(triangles);
    cudaMemcpy(devKdTree, &hostTree, sizeof(KdTree<TREE_DEPTH>), cudaMemcpyHostToDevice);

    cudaFree(devMaterials); CUDA_CHECK
    cudaMalloc(&devMaterials, sizeof(mcMaterial) * materialCount); CUDA_CHECK
    cudaMemcpy(devMaterials, materials, sizeof(mcMaterial) * materialCount, cudaMemcpyHostToDevice); CUDA_CHECK
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


void cudaTracePixels(std::vector<QueuedPixel> &pixels, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    QueuedPixel* devPixels;
    cudaMalloc(&devPixels, sizeof(QueuedPixel) * pixels.size()); CUDA_CHECK
    cudaMemcpy(devPixels, pixels.data(), sizeof(QueuedPixel) * pixels.size(), cudaMemcpyHostToDevice); CUDA_CHECK

    size_t num_passes = 0;
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        num_passes = std::max(size_t(pixels[i].samples + 10 - 1) / 10, num_passes);
    }

    float3* devResults;
    cudaMalloc(&devResults, sizeof(float3) * pixels.size()); CUDA_CHECK

    assert(pixels.size() % CUDA_BLOCK_SIZE == 0);
    float3* hostResults = new float3[pixels.size()];

    for (int pass = 0; pass < num_passes; ++pass)
    {
        //tracePixels<<<pixels.size() / cudaBlockSize(), cudaBlockSize()>>>(devPixels, devResults, devMaterials, devKdTree, devCamera, rand() << 16 | rand()); CUDA_CHECK
        tracePixels<<<pixels.size() / cudaBlockSize(), cudaBlockSize()>>>(devPixels, devResults, devMaterials, devTriangles, devTriangleCount, devCamera, rand() << 16 | rand()); CUDA_CHECK

        cudaMemcpy(hostResults, devResults, sizeof(float3) * pixels.size(), cudaMemcpyDeviceToHost); CUDA_CHECK

        for (size_t i = 0; i < pixels.size(); ++i)
        {
            QueuedPixel& pixel = pixels[i];
            size_t index = pixel.x + pixel.y * imageWidth;
            colorMap[index] += toACG3(hostResults[i]);

            const int samples = std::min(pixel.samples, 10);
            pixel.samples -= samples;
            sampleCounter[index] += samples;
        }
    }

    cudaFree(devPixels); CUDA_CHECK
    cudaFree(devResults); CUDA_CHECK

    delete[] hostResults;
}

